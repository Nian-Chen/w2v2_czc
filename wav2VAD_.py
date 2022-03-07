#-*-coding:utf-8-*-
import math
import librosa
import torch
import numpy as np
import os
from tqdm import tqdm
import glob
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    PreTrainedTokenizerBase,
)
from typing import Optional, Tuple, List
from pyctcdecode import build_ctcdecoder
def get_chunck_start_end(speech_length,chunck_dur):
    '''
    input:int
    output:[tuple(chunck_start,chunck_end),,,,]
    '''
    # 总采样点数
    total_samples = speech_length
    # chunck大小设置为5min
    chunck_size = int(chunck_dur*60*16000)
    # chunck数量，int，此例为4
    print(f"total_samples = {total_samples}")
    print(f"chunck_size = {chunck_size}")
    chuncks = math.ceil(total_samples / chunck_size)
    # 最后一个chunck的帧数，要考虑刚好整除的情况，此例为1000
    last_chunck_samples = total_samples % chunck_size if total_samples % chunck_size!=0 else chunck_size
    print(f"last_chunck_samples:{last_chunck_samples}")
    # 每个chunck的起始帧
    chuncks_start = list(range(0,total_samples,chunck_size))
    print(f"chuncks_start = {chuncks_start}")
    # 每个chunck的结束帧等于每个chunck的起始帧加上chunck_size（除了最后一个chunck要取决于last_chunck_frames）
    chuncks_end = (np.ones(chuncks,dtype=int)*chunck_size+np.array(chuncks_start,dtype=int)).tolist()
    print(f"chuncks_end = {chuncks_end}")
    # 针对最后一个chunck做修改（最后一个chunck要取决于last_chunck_frames）
    chuncks_end[-1] = chuncks_start[-1] + last_chunck_samples
    print(f"chuncks_end = {chuncks_end}")
    if last_chunck_samples<=400:
        # last_chunck_samples小于400（25ms）会导致前向计算失败,将剩下这些samples加入最后一个chunck
        chuncks=chuncks-1
        chuncks_start.pop()
        chuncks_end.pop()
        chuncks_end[-1] = chuncks_end[-1] + last_chunck_samples
    #
    assert (chuncks==len(chuncks_start)==len(chuncks_end)),f"length of chuncks_end/start not equal to chuncks"
    return(list(zip(chuncks_start,chuncks_end)))
# print(get_chunck_start_end(8*60*16000+300,4))


def get_kenlm_decoder(
        vocabulary: List[str],
        lm_path: Optional[str] = None,
        alpha: float = 0.0,
        beta: float = 0.0,
):
    # https://github.com/kensho-technologies/pyctcdecode.git
    kenlm_decoder = build_ctcdecoder(
        labels=vocabulary,
        kenlm_model_path=lm_path,
        alpha=alpha,
        beta=beta,
    )
    return kenlm_decoder
def main():
    def get_pad_probs(speech):
        '''
        input:speech (L,)
        output:pad_logits tensor(1,L); pad_probs tensor(1,L)
        '''
        features = processor(speech, return_tensors="pt", padding="longest", sampling_rate=16000)
        input_values = features.input_values.to(device)
        #     input_values = processor(batch["input_values"], return_tensors="pt", padding="longest",sampling_rate=16000).input_values
        with torch.no_grad():
            logits = model(input_values).logits.cpu()
        # 输出原logits
        pad_logits = logits
        pad_probs = torch.nn.functional.softmax(logits, dim=-1, dtype=torch.float32)[:, :, 0]
        print(f"frames of chunck = {pad_probs.shape[1]}")
        predicted_ids = torch.max(logits, dim=-1)[1]
        transcription = processor.batch_decode(predicted_ids)
        print(transcription)
        #     print(pad_logits,"\n",pad_probs,"\n",pad_ids)
        return pad_logits, pad_probs  # ,pad_ids
    def get_isspeech_list(logits,sil_frames_th):
        '''
        input:logits tensor(L,V)
        output:isspeech_list list(tuple)
        '''
        aligh_list = kenlm_decoder.decode_beams(logits[0].cpu().numpy(), beam_width=20)[0][2]
        isspeech_list = []
        for index, i in enumerate(aligh_list):
            if index == 0:
                last_end = i[1][1]
                frame_s = i[1][0]
            else:
                if (i[1][0] - last_end) >= sil_frames_th:
                    # print(i[1][0] - last_end)
                    frame_e = aligh_list[index - 1][1][-1]
                    isspeech_list.append((round((frame_s - 1) * 0.02, 2), round((frame_e - frame_s) * 0.02 + 0.025, 2)))
                    frame_s = aligh_list[index][1][0]
                #             frame_s = aligh[index-1][-1]
                #             frame_e = aligh[index][0]
                #             result.append(((frame_s - 1)*0.02,(frame_e - frame_s)*0.02 + 0.025 ))
                last_end = i[1][1]
        return isspeech_list # ,pad_ids
    device = "cuda:1"
    speech_files = glob.glob(os.path.join("/tsdata/diarization/voxconverse21_duke/DEV/audio/", '*.wav'))[-5:]
    speech_files = ["/tsdata/diarization/voxconverse21_duke/DEV/audio/wemos.wav"]
    print(f"num_files = {len(speech_files)}")
    # print(speech_files[1].split(".")[0].split("/")[-1])
    model_path = "/data2_from_58175/huggingface/models/wav2vec2_gpt2/encoder"
    model = Wav2Vec2ForCTC.from_pretrained(model_path).to(device)
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    vocab_dict = processor.tokenizer.get_vocab()
    sort_vocab = sorted((value, key) for (key,value) in vocab_dict.items())
    vocabulary = [x[1] for x in sort_vocab]
    kenlm_decoder = get_kenlm_decoder(
                vocabulary=vocabulary,
                )
    # 音频太长，一次性处理会导致显存不足，需要分块处理
    chunck_dur = 4
    score_dict = {}
    sil_frames_th = 100
    score_dict_save_name = f"/tsdata/wav2vad/wemos_wav2bsvad_4min_{sil_frames_th}.npy"
    for index,speech_name in tqdm(enumerate(speech_files),total =len(speech_files),leave=True):
        speech,_ = librosa.load(speech_name,sr=16000)
        # 名称去掉.wav作为dict的key
        speech_key = speech_name.split(".")[0].split("/")[-1]
        print(f"speech_key={speech_key}")
        speech_length = speech.shape[0]
        # 超过chunck_size则切为多个chunck，依次识别后再将logits拼回
        chunck_dur = chunck_dur
        chunck_size = chunck_dur*60*16000
        print(f"chunck_size={chunck_size/16000/60}")
        if speech_length <= chunck_size:
            print(f"speech_length {round(speech_length/16000/60,1)}<={chunck_size/16000/60}min, only one chunk")
            pad_logits,pad_probs = get_pad_probs(speech)
            # print(f"pad_logits.shape={pad_logits.shape}")
            # print(f"pad_probs.shape={pad_probs.shape}")
    #         print(f"pad_ids.shape={pad_ids.shape}")
        else:
            print(f"speech_length {round(speech_length/16000/60,1)}>{chunck_size/16000/60}min, cut into chunks")
            chuncks_start_end = get_chunck_start_end(speech_length,chunck_dur)
            # print(f"chuncks_start_end={chuncks_start_end}")
            pad_logits_list = []
            # pad_probs_list = []
    #         pad_ids_list = []
            for chunck in chuncks_start_end:
                pad_logits_probs = get_pad_probs(speech[chunck[0]:chunck[1]])
                pad_logits_list.append(pad_logits_probs[0])
                # pad_probs_list.append(pad_logits_probs[1])
                # torch.cuda.empty_cache()
    #             pad_ids_list.append(pad_logits_probs[2])
    #         pad_probs = torch.cat(pad_probs_list,1)
            pad_logits = torch.cat(pad_logits_list,1)
    #         pad_ids = torch.cat(pad_ids_list,1)
    #         print(f"pad_probs.shape={pad_probs.shape}")
    #         print(f"pad_logits.shape={pad_logits.shape}")
    #         print(f"pad_ids.shape={pad_ids.shape}")
    #         print(f"pad_probs={pad_probs}")
    #         print(f"pad_logits={pad_logits}")
    #         print(f"pad_ids{pad_ids}")
    #     score_dict[speech_key]=(pad_logits,pad_probs,pad_ids)
        logits = pad_logits
        isspeech_list = get_isspeech_list(logits,sil_frames_th)
        # print(speech_key,":\n",isspeech_list)
        score_dict[speech_key]=(isspeech_list)
    np.save(score_dict_save_name, score_dict)
if __name__ == "__main__":
    main()