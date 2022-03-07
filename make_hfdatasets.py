import argparse
import pandas as pd
import os
from datasets import load_dataset, load_from_disk
from transformers import Wav2Vec2Processor
os.environ["TRANSFORMERS_CACHE"] = "/data2_from_58175/huggingface/transformers"
os.environ["HF_DATASETS_CACHE"] = "/data2_from_58175/huggingface/datasets"
os.environ["HF_METRICS_CACHE"] = "/data2_from_58175/huggingface/metrics"
os.environ["HF_HOME"] = "/data2_from_58175/huggingface"
# os.environ["TMPDIR"] = "/data2_from_58175/tmp"
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="make dataset for w2v2_finetuning")
    parser.add_argument("-f_train", "--input_train_file", type=str, default=None, help="the train_dataset file generating from kaldi_files")
    parser.add_argument("-f_dev", "--input_dev_file", type=str, default=None, help="the dev_dataset file generating from kaldi_files")
    parser.add_argument("-f_test", "--input_test_file", type=str, default=None, help="the test_dataset file generating from kaldi_files")
    parser.add_argument("-m", "--segments_mode", action='store_true', default=False, help="segmets or no_segments")
    parser.add_argument("-ch", "--is_ch", action='store_true', default=False, help="chinese or english, True means chinese")
    # parser.add_argument("-p_processor", "--processor_path", type=str, help="the path of processor")
    parser.add_argument("-p_trim", "--trim_audio_dir", type=str, default=None, help="the path of trim_audio")
    parser.add_argument("-maxl", "--max_length", type=str, default=100.0, help="filter by max_length of audio")
    parser.add_argument("-minl", "--min_length", type=str, default=0.0, help="filter by min_length of audio")
    parser.add_argument("-p_save", "--save_path", type=str, default=None, help="the path to save datasets")

    args = parser.parse_args()
    input_train_file = args.input_train_file
    input_dev_file = args.input_dev_file
    input_test_file = args.input_test_file
    segments_mode = args.segments_mode
    trim_audio_dir = args.trim_audio_dir
    max_length = args.max_length
    min_length = args.min_length
    if segments_mode :
        if trim_audio_dir is None:
            raise ValueError(
                f"trim_file_path is required in segments_mode"
            )
    is_ch = args.is_ch

    # processor_path = args.processor_path
    # data/tmp/dataset
    save_path = args.save_path
    # data/hf_datasets
    print(f"args = {args}")

    def get_csv(file,segments_mode,split,is_ch):
        if segments_mode:
            column_names = ['id','file','seg_start','seg_end','length'] if split == "test" else ['id','file','seg_start','seg_end','length','text']
        else:
            # column_names = ['id','file','length'] if split == "test" else ['id','file','length','text']
            column_names = ['id','file','length','text']
        df = pd.read_csv(file,delimiter="\t",names=column_names)
        # if split != "test":
        if is_ch:
            # 将中文字用空格隔开，使其带有空格token
            df[u'text'] = df[u'text'].apply(lambda x: " ".join(x))
        else:
            # 默认非中文即英文，做大写规范
            df[u'text'] = df[u'text'].apply(lambda x :x.upper())
        df.to_csv(f"{file}.csv", encoding='utf_8_sig', index=False)
        return df
    csv_train_dataset = get_csv(input_train_file, segments_mode=segments_mode, split="train", is_ch=is_ch) if input_train_file is not None else None
    csv_dev_dataset = get_csv(input_dev_file, segments_mode=segments_mode, split="dev", is_ch=is_ch) if input_dev_file is not None else None
    csv_test_dataset = get_csv(input_test_file, segments_mode=segments_mode, split="test", is_ch=is_ch) if input_test_file is not None else None
    csv_train_dataset_path = f"{input_train_file}.csv"
    csv_dev_dataset_path = f"{input_dev_file}.csv"
    csv_test_dataset_path = f"{input_test_file}.csv"

    if (input_train_file and input_dev_file and input_test_file) is not None:
        final_dataset = load_dataset('csv', data_files={"train": csv_train_dataset_path,
                                                        "dev": csv_dev_dataset_path,
                                                        "test": csv_test_dataset_path},cache_dir=save_path)
    elif (input_train_file and input_dev_file) is not None:
        final_dataset = load_dataset('csv', data_files={"train": csv_train_dataset_path,
                                                        "dev": csv_dev_dataset_path,},cache_dir=save_path)
    elif input_train_file is not None:
        final_dataset = load_dataset('csv', data_files={"train": csv_train_dataset_path},cache_dir=save_path)
    elif input_dev_file is not None:
        final_dataset = load_dataset('csv', data_files={"dev": csv_dev_dataset_path},cache_dir=save_path)
    elif input_test_file is not None:
        final_dataset = load_dataset('csv', data_files={"test": csv_test_dataset_path},cache_dir=save_path)
    else:
        raise ValueError(
            f"no dataset is available"
        )
    # processor = Wav2Vec2Processor.from_pretrained(processor_path)
    def map2segfile(batch):
        # batch["labels"] = processor.tokenizer(batch["text"])
        # 如果采用segments做的数据集，需要把数据根据uttid离线裁剪出来存到trim_file_path，这样便于训练，所以需要更新batch["file"]
        # trim_file_path = "/data2/fisher_swbd_nodup_trim/"
        batch["file"] = trim_audio_dir+"/"+batch["id"]+".flac" if trim_audio_dir else batch["id"]+".flac"
        return batch
    final_dataset = final_dataset.map(map2segfile, keep_in_memory=True) if segments_mode else final_dataset
    if input_test_file is None:
        final_dataset = final_dataset.filter(lambda batch: float(min_length)<=batch["length"]<=float(max_length))
    final_dataset.save_to_disk(save_path)