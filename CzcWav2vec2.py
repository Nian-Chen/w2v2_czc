from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM, PretrainedConfig
from transformers import Wav2Vec2Processor, Wav2Vec2Config, EncoderDecoderConfig, EncoderDecoderModel, GPT2Config, \
    Wav2Vec2PreTrainedModel
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2ForPreTrainingOutput, Wav2Vec2ForPreTraining, \
    _compute_mask_indices, _sample_negative_indices, Wav2Vec2ForSequenceClassification, Wav2Vec2Model, Wav2Vec2BaseModelOutput, \
    Wav2Vec2EncoderLayerStableLayerNorm, Wav2Vec2EncoderStableLayerNorm, Wav2Vec2PositionalConvEmbedding
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List
from torch import nn
from transformers.modeling_outputs import Seq2SeqLMOutput, CausalLMOutput, CausalLMOutputWithCrossAttentions
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer, Wav2Vec2ForCTC, GPT2LMHeadModel, \
    Wav2Vec2Model
from torch.nn import CrossEntropyLoss, MSELoss
from speechbrain.lobes import features
from speechbrain.processing.features import STFT,spectral_magnitude
from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from torchaudio import sox_effects
from transformers.file_utils import ModelOutput
import scipy.io as sio
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
def _compute_mask_indices_random(
    shape: Tuple[int, int],
    mask_prob: float,
    attention_mask: Optional[torch.LongTensor] = None,
    min_masks: int = 1,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape. Used to implement `SpecAugment: A Simple Data Augmentation Method for
    ASR <https://arxiv.org/abs/1904.08779>`__. Note that this method is not optimized to run on TPU and should be run
    on CPU as part of the preprocessing during training.

    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_length: size of the mask
        min_masks: minimum number of masked spans
    """
    batch_size, sequence_length = shape
    epsilon = np.random.rand(1).item()

    def compute_num_masked_span(input_length):
        """Given input length, compute how many spans should be masked"""
        num_masked_span = int(mask_prob * input_length + epsilon)
        num_masked_span = max(num_masked_span, min_masks)

        # make sure num masked indices <= sequence_length
        num_masked_span = sequence_length if num_masked_span > sequence_length else num_masked_span
        return num_masked_span

    # compute number of masked spans in batch
    input_lengths = (
        attention_mask.sum(-1).detach().tolist()
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]
    )
    # print(f"input_lengths = {input_lengths}")
    # SpecAugment mask to fill
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=np.bool)
    spec_aug_mask_idxs = []

    max_num_masked_span = compute_num_masked_span(sequence_length)
    
    for input_length in input_lengths:
        # compute num of masked spans for this input
        num_masked_span = compute_num_masked_span(input_length)
        # print(f"num_masked_span = {num_masked_span}")
        # get random indices to mask
        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length), num_masked_span, replace=False
        )
        # print(f"spec_aug_mask_idx = {spec_aug_mask_idx}")

        # pick first sampled index that will serve as a dummy index to pad vector
        dummy_mask_idx = spec_aug_mask_idx[0]

        spec_aug_mask_idx = np.concatenate(
            [spec_aug_mask_idx, np.ones(max_num_masked_span - num_masked_span, dtype=np.int32) * dummy_mask_idx]
        )
        spec_aug_mask_idxs.append(spec_aug_mask_idx)

    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)

    # scatter indices to mask
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    return spec_aug_mask

def get_czc_scheduler(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, lr_end=1e-7, power=2.0, last_epoch: int = -1
):
    lr_init = optimizer.defaults["lr"]
    assert lr_init > lr_end, f"lr_end ({lr_end}) must be be smaller than initial lr ({lr_init})"
    def lr_lambda(current_step):
        num_last_steps = 0.6 * num_training_steps
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step > num_last_steps:
            lr_range = lr_init - lr_end
            decay_steps = num_training_steps - num_last_steps
            pct_remaining = 1 - (current_step - num_last_steps) / decay_steps
            decay = lr_range * pct_remaining ** power + lr_end
            return decay / lr_init  # as LambdaLR multiplies by lr_init
        elif current_step > num_training_steps:
            return lr_end / lr_init  # as LambdaLR multiplies by lr_init 不是return 1
        else:
            return 1  # as LambdaLR multiplies by lr_init 不是return 1
        
    return LambdaLR(optimizer, lr_lambda, last_epoch)
def get_w2v2_scheduler(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, last_epoch: int = -1
):
    def lr_lambda(current_step):
        num_last_steps = 0.5 * num_training_steps
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step > num_last_steps:
            return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_last_steps))
        )        
        return 1

    return LambdaLR(optimizer, lr_lambda, last_epoch)
class Wav2Vec2ForPreTrainingOutput_mfcc_auxiliary(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    projected_states: torch.FloatTensor = None
    mfcc_pred: torch.FloatTensor = None
    mfcc_target: torch.FloatTensor = None
    projected_quantized_states: torch.FloatTensor = None
    codevector_perplexity: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    contrastive_loss: Optional[torch.FloatTensor] = None
    diversity_loss: Optional[torch.FloatTensor] = None
    auxiliary_loss: Optional[torch.FloatTensor] = None


class Wav2Vec2ForPreTrainingOutput_mfcc(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    mfcc_pred: torch.FloatTensor = None
    mfcc_target: torch.FloatTensor = None
    mae_decoder_input: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class GPT2_Decoder(GPT2LMHeadModel):
    def LabelSmoothingLoss(
            self,
            logits,
            labels,
            smoothing: float = 0.1,
            padding_idx: int = -100,
            vocab_size: int = 32,
            normalize_length: bool = False,
            decoder_teacher_logits=None,
            pseudo_rows=None,
            decoder_entropy_loss=False,
            negative_sampling_loss=False,
            negative_sampling_loss_add=False,
            reduction='sum',
    ):
        # print("doing label smoothing")
        criterion = nn.KLDivLoss(reduction="none")
        confidence = 1.0 - smoothing
        assert logits.size(2) == vocab_size
        batch_size = logits.size(0)
        logits = logits.view(-1, vocab_size)
        original_labels = labels
        labels = labels.view(-1)
        true_dist = torch.zeros_like(logits)
        # 先用0.1/(32-1)填满，再将对应的标签位置为0.9
        true_dist.fill_(smoothing / (vocab_size - 1))
        # padding_index=-100用0代替，否则计算loss会报错，选用0是因为它在labels中并不会出现
        ignore = labels == padding_idx  # (B,)
        # 默认loss为sum/batch_size，若normalize_length = True则sum/tokens
        total = len(labels) - ignore.sum().item()
        labels = labels.masked_fill(ignore, 0)  # avoid -1 index
        true_dist.scatter_(1, labels.unsqueeze(1), confidence)
        # print(logits.shape)
        # print(true_dist.shape)
        # print(f"pseudo_rows in kl_loss = {pseudo_rows}")
        # print(pseudo_rows)
        pseudo_labels_mask = torch.zeros_like(original_labels).bool()
        pseudo_labels_mask[pseudo_rows] = True
        pseudo_labels_mask = pseudo_labels_mask.view(-1)
        pseudo_weight = None
        true_label_weight = 1.0
        # pseudo_weight = 10.0
        if (decoder_teacher_logits is None and decoder_entropy_loss is False and negative_sampling_loss is False):
            # 正常计算loss或者不存在伪标签
            kl = criterion(torch.log_softmax(logits, dim=-1), true_dist).masked_fill(ignore.unsqueeze(1), 0)  # .sum()
            #################分配伪标签权重：
            if pseudo_weight != None:
                # print(f"kl = {kl.sum()}")
                kl_weight_true = kl.masked_fill((~pseudo_labels_mask).unsqueeze(1), true_label_weight)
                kl_weight_pseudo = kl_weight_true.masked_fill((pseudo_labels_mask).unsqueeze(1), pseudo_weight)
                kl = (kl_weight_pseudo * kl).sum()
                # print(f"kl after weighted= {kl}")
                return kl
            else:
                # print(kl)
                if reduction == "mean":
                    return kl.sum() / total
                return kl.sum()
        else:
            # print(f"pseudo_labels_mask = {pseudo_labels_mask}")
            # print(f"pseudo_labels_mask.shape = {pseudo_labels_mask.shape}")
            if decoder_teacher_logits:
                decoder_teacher_logits = decoder_teacher_logits.view(-1, vocab_size)
                assert decoder_teacher_logits.shape == logits.shape  # and (decoder_teacher_logits!=logits).sum()!=0
            # 得到的kl loss是二维的，所以mask需要unsqueeze(1)
            # print((~pseudo_labels_mask).sum())
            if (~pseudo_labels_mask).sum() == 0:
                # print("all labels are pseudo_labels")
                # 全为伪标签，则全计算soft loss
                if decoder_entropy_loss:
                    # 伪标签数据计算熵值，使熵最小化
                    logits_onehot = F.softmax((logits / 1e-2), dim=-1)
                    # print(logits_onehot)
                    # print(logits_onehot.sum())
                    kl_soft = criterion(torch.log_softmax(logits, dim=-1), logits_onehot) * 0.1
                    # print(kl_soft.sum())
                    # logits_onehot中温度设置太小会导致nan
                    print("kl_soft = 0") if kl_soft.sum() == 0 else None
                elif decoder_teacher_logits:
                    kl_soft = criterion(torch.log_softmax(logits, dim=-1),
                                        torch.softmax(decoder_teacher_logits, dim=-1))
                elif negative_sampling_loss:
                    # 在原来klloss基础上加
                    kl_all = criterion(torch.log_softmax(logits, dim=-1), true_dist).masked_fill(ignore.unsqueeze(1),
                                                                                                 0).sum()
                    # 只求和每个token输出中较低log概率，越小越好
                    probs = torch.softmax(logits, dim=-1)
                    # [[0.98,0.01,0.01],[0.51,0.42,0.03]]
                    threshold = 0.001
                    negative_mask = probs > threshold
                    # [[1,0,0],[1,1,0]
                    probs = 1.0 - probs.masked_fill(negative_mask, 0).sum(-1)
                    # [1,1] - [0.02,0.03] = [0.98,0.97]
                    probs = probs.masked_fill(ignore, 1)
                    # print(f"probs = {probs}")
                    # 用1来mask，因为接下来要算log，1对应log为0，则不影响
                    negative_sampling_loss = -probs.log().sum()
                    # print(f"negative_sampling_loss = {negative_sampling_loss}")
                    return negative_sampling_loss + kl_all
                return kl_soft.masked_fill(ignore.unsqueeze(1), 0).sum()
            else:
                # 对于伪标注数据，是否使用最小化熵loss
                # batch不全为伪标签样本
                # print("has both true_labels and pseudo_labels")
                # print(f"pseudo_rows in kl_loss = {pseudo_rows}")
                kl_all = criterion(torch.log_softmax(logits, dim=-1), true_dist).masked_fill(ignore.unsqueeze(1),
                                                                                             0).sum()
                kl_hard = criterion(torch.log_softmax(logits, dim=-1), true_dist).masked_fill(
                    pseudo_labels_mask.unsqueeze(1), 0).masked_fill(ignore.unsqueeze(1), 0).sum()
                # print(f"kl_hard.sum() = {kl_hard.sum()}")
                if decoder_entropy_loss:
                    logits_onehot = F.softmax((logits / 1e-2), dim=-1)
                    kl_soft = criterion(torch.log_softmax(logits, dim=-1), logits_onehot).masked_fill(
                        (~pseudo_labels_mask).unsqueeze(1), 0) * 0.1
                    print("kl_soft = 0") if kl_soft.sum() == 0 else None
                    # print(kl_soft.sum())
                elif decoder_teacher_logits:
                    kl_soft = criterion(torch.log_softmax(logits, dim=-1),
                                        torch.softmax(decoder_teacher_logits, dim=-1)).masked_fill(
                        (~pseudo_labels_mask).unsqueeze(1), 0)
                elif negative_sampling_loss:
                    # 只求和每个token输出中较低概率，然后用1向量减去它，取-log
                    probs = torch.softmax(logits, dim=-1)
                    # [[0.98,0.01,0.01],[0.51,0.42,0.03]]
                    threshold = 0.001
                    negative_mask = probs > threshold
                    # [[1,0,0],[1,1,0]
                    probs = 1.0 - probs.masked_fill(negative_mask, 0).sum(-1)
                    # [1,1] - [0.02,0.03] = [0.98,0.97]
                    probs = probs.masked_fill(ignore, 1)  # .masked_fill(~pseudo_labels_mask, 1)
                    # print(f"probs = {probs}")
                    # 用1来mask，因为接下来要算log，1对应log为0，则不影响
                    negative_sampling_loss = -probs.log().sum()
                    # print(f"negative_sampling_loss = {negative_sampling_loss}")
                    return kl_all + negative_sampling_loss
                kl = kl_hard + kl_soft
                return kl.masked_fill(ignore.unsqueeze(1), 0).sum()
        # denom = total if normalize_length else batch_size
        # 返回sum
        return kl

    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            decoder_teacher_logits=None,
            pseudo_rows=None,
            decoder_entropy_loss=False,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)
        do_label_smoothing = True
        loss = None
        lsm_reduction = "sum" if self.config.add_cross_attention else "mean"
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            if decoder_teacher_logits is not None:
                decoder_teacher_logits = decoder_teacher_logits[..., :-1, :].contiguous()
            if do_label_smoothing:
                loss = self.LabelSmoothingLoss(logits=shift_logits,
                                               labels=shift_labels,
                                               smoothing=0.1,
                                               padding_idx=-100,
                                               vocab_size=self.config.vocab_size,
                                               decoder_teacher_logits=decoder_teacher_logits,
                                               pseudo_rows=pseudo_rows,
                                               reduction=lsm_reduction,
                                               decoder_entropy_loss=decoder_entropy_loss)
                # loss = self.LabelSmoothingLoss(logits = shift_logits,
                # labels = shift_labels,
                # smoothing = 0.1,
                # padding_idx = -100,
                # vocab_size = 50257,
                # decoder_teacher_logits = decoder_teacher_logits,
                # pseudo_rows = pseudo_rows,
                # reduction = 'mean',
                # decoder_entropy_loss = decoder_entropy_loss)
                # print(loss.item())
            else:
                # Shift so that tokens < n predict n
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                ###########################################
                # labels被shift，那么redution="mean"求取的并非batch mean，而是token mean，故与wav2vec2保持一致，此步返回sum
                loss_fct = CrossEntropyLoss(reduction='mean')
                # gpt2微调计算loss时没考虑pad，因为labels是等长的（拼接到block_size）
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                # nan_mask = torch.isnan(loss)
                # if sum(nan_mask) > 0:
                # print(hidden_states)
                # print(lm_logits)
                # print(attention_mask)
                # print(encoder_attention_mask)
                # print(labels)
                # if sum(~nan_mask) == 0:
                # print("!"*20,"cross_entopy_loss all is nan","!"*20)
                # loss = torch.tensor(0.0,requires_grad=True)
                # else:
                # print("!"*20,"cross_entopy_loss has nan","!"*20)
                # loss = loss.masked_select(~nan_mask).sum()
                # print(loss)
                # print(torch.isnan(loss))
                # else:
                # loss = loss.sum()
            do_predict_pseudo = False
            if do_predict_pseudo:
                # set_trace()
                eos_index = (labels == self.eos_id).nonzero()
                eos_index = eos_index.transpose(0, 1).tolist()
                # 取出eos对应的hidden_states
                logits_eos = hidden_states[eos_index].contiguous()
                pred_pseudo_loss_func = nn.BCELoss(reduction='sum')
                sigmoid = nn.Sigmoid()
                pred_pseudo_probs = sigmoid(self.pred_pseudo_head(logits_eos)).squeeze(1)
                pred_pseudo_targets = torch.ones(labels.shape[0])
                # 0-1可代表置信度，真实标签置信度为1
                pred_pseudo_targets[pseudo_rows] = 0
                pred_pseudo_loss = pred_pseudo_loss_func(pred_pseudo_probs, pred_pseudo_targets)
                # 由它们预测样本是否为伪标注
                loss = loss + pred_pseudo_loss
                output = (lm_logits,) + transformer_outputs[1:]
                return ((loss,) + output) if loss is not None else output

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )


class Wav2vec2_Gpt2(EncoderDecoderModel):
    def pad_list(self, xs: List[torch.Tensor], pad_value: int):
        """Perform padding for the list of tensors.

        Args:
            xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
            pad_value (float): Value for padding.

        Returns:
            Tensor: Padded tensor (B, Tmax, `*`).

        Examples:
            >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
            >>> x
            [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
            >>> pad_list(x, 0)
            tensor([[1., 1., 1., 1.],
                    [1., 1., 0., 0.],
                    [1., 0., 0., 0.]])

        """
        n_batch = len(xs)
        max_len = max([x.size(0) for x in xs])
        pad = torch.zeros(n_batch, max_len, dtype=xs[0].dtype, device=xs[0].device)
        pad = pad.fill_(pad_value)
        for i in range(n_batch):
            pad[i, :xs[i].size(0)] = xs[i]

        return pad

    def add_sos_eos(self, ys_pad: torch.Tensor, sos: int, eos: int,
                    ignore_id: int) -> torch.Tensor:
        """Add <sos> and <eos> labels.

        Args:
            ys_pad (torch.Tensor): batch of padded target sequences (B, Lmax)
            sos (int): index of <sos>
            eos (int): index of <eeos>
            ignore_id (int): index of padding

        Returns:
            ys_in (torch.Tensor) : (B, Lmax + 1)
            ys_out (torch.Tensor) : (B, Lmax + 1)

        Examples:
            >>> sos_id = 1
            >>> eos_id = 2
            >>> ignore_id = -100
            >>> ys_pad
            tensor([[ 2,  3,  4,    5,    6],
                    [ -1, 7,  8, -100, -100],
                    [ 9, 10, 11, -100, -100]], dtype=torch.int32)
            >>> out=add_sos_eos(ys_pad, sos_id , eos_id, ignore_id)
            >>> ys_in
            tensor([[ 1, 2,  3,    4,    5,    6],
                    [ 1, 7,  8, -100, -100, -100],
                    [ 1, 9, 10, -100, -100, -100]], dtype=torch.int32)
            >>> ys_out
            tensor([[ 1, 2,  3, 4,    5,    6],
                    [ 1, 7,  8, 2, -100, -100],
                    [ 1, 9, 10, 2, -100, -100]], dtype=torch.int32)
        """
        # 实时伪标签以pseudo_label_id=-1开头，所以在add_sos_eos需要把-1删去
        pseudo_label_id = -1
        _sos = torch.tensor([sos],
                            dtype=torch.long,
                            requires_grad=False,
                            device=ys_pad.device)
        _eos = torch.tensor([eos],
                            dtype=torch.long,
                            requires_grad=False,
                            device=ys_pad.device)
        ys = [y[(y != ignore_id) * (y != pseudo_label_id)] for y in ys_pad]  # parse padded ys
        #     ys_in = [torch.cat([_sos, y], dim=0) for y in ys]
        #     ys_out = [torch.cat([y, _eos], dim=0) for y in ys]
        ys_out = [torch.cat([_sos, y, _eos], dim=0) for y in ys]
        return self.pad_list(ys_out, ignore_id)

    def freeze_decoder(self):
        for param in self.decoder.parameters():
            param.requires_grad = False

    def forward(
            self,
            input_values=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=True,
            return_dict=None,
            forward_only_encoder=None,
            decoder_teacher_logits=None,
            encoder_teacher_logits=None,
            decoder_entropy_loss=False,
            encoder_entropy_loss=False,
            **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_"):]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }
        # self.encoder.wav2vec2输出的last_hidden_state与encoder_outputs.hidden_states[-1]一致
        #         wav2vec2_hidden_state = self.encoder.wav2vec2(
        #             input_values=input_values,
        #             attention_mask=attention_mask,
        #             output_attentions=output_attentions,
        #             output_hidden_states=False,
        #             return_dict=return_dict,
        #         )[0]
        #         print(wav2vec2_hidden_state)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_values=input_values,
                labels=labels,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                encoder_teacher_logits=encoder_teacher_logits,
                encoder_entropy_loss=encoder_entropy_loss,
                **kwargs_encoder,
            )

        #         print(len(encoder_all_hidden_states))
        encoder_logits = encoder_outputs.logits
        encoder_hidden_states = encoder_outputs.hidden_states[-1]
        #         encoder_hidden_states.retain_grad()
        #         print(encoder_hidden_states.shape)
        self.encoder_hidden_states = encoder_hidden_states
        #         print(self.encoder_hidden_states.requires_grad)

        # 传给encoder的labels不能直接传给decoder的input_ids及labels
        # 对于decoder_labels，对labels进行操作: labels中（非-100）首尾添加<bos>、<eos>
        # 对于decoder_input_ids，在decoder_labels基础上，需要用0替代-100，否则nn.embedding(-100)越界
        # 用0取代后pad的token不会对句子造成影响（attention_mask），一般输入token中没有0
        # 在decoder_labels基础上取dec_input_attention_mask
        # print(f"labels={labels}") if labels is not None else None
        #         print(f"attention_mask.shape={attention_mask.shape}")
        assert (self.decoder.config.bos_token_id == self.encoder.config.bos_token_id) and (
                self.decoder.config.eos_token_id == self.encoder.config.eos_token_id)
        assert self.decoder.config.vocab_size == self.encoder.config.vocab_size
        sos_id = self.decoder.config.bos_token_id
        eos_id = self.decoder.config.eos_token_id
        #         print(f"sos_id={sos_id}")
        #         print(f"eos_id={eos_id}")
        ignore_id = -100
        if not forward_only_encoder:
            pseudo_mask = labels == -1
            # 得到伪标签样本在当前batch的行索引值
            pseudo_rows = pseudo_mask.nonzero()[:, 0]  # .tolist()
            pseudo_rows = None if pseudo_rows == [] else pseudo_rows
            # print(f"pseudo_rows = {pseudo_rows}")
            # pseudo_rows [1,5,6,7,8]代表第1，5，6，7，8为伪标签样本，为空[]则代表不存在伪标签样本
            # 传给decoder，用于分开二者loss的计算，一部分使用hard，一部分使用soft
            dec_labels = self.add_sos_eos(labels, sos_id, eos_id, ignore_id)
            dec_input_attention_mask = dec_labels.ne(-100)
            dec_input_ids = dec_labels.masked_fill(~dec_input_attention_mask, 0)
            # print(f"dec_labels={dec_labels}")
        #             print(f"dec_input_attention_mask={dec_input_attention_mask}")
        # print(f"dec_input_ids={dec_input_ids}")
        # 不能用传入的attention_mask，那是音频采样点级别的，在cross_attention时需要帧级别的attention_mask
        with torch.no_grad():
            wav2vec2 = self.encoder.wav2vec2
            extract_features = wav2vec2.feature_extractor(input_values).transpose(1, 2)
            output_lengths = wav2vec2._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)
            enc_frame_attention_mask = torch.zeros(
                extract_features.shape[:2], dtype=extract_features.dtype, device=extract_features.device
            )
            # these two operations makes sure that all values
            # before the output lengths indices are attended to
            enc_frame_attention_mask[
                (torch.arange(enc_frame_attention_mask.shape[0], device=extract_features.device), output_lengths - 1)
            ] = 1
            enc_frame_attention_mask = enc_frame_attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        #         print(f"enc_frame_attention_mask.shape:{enc_frame_attention_mask.shape}")
        #         print(f"enc_frame_attention_mask:{enc_frame_attention_mask}")
        #         self.encoder_attention_mask = enc_frame_attention_mask
        #         self.encoder_hidden_states = encoder_hidden_states
        # Decode
        vocab_size = self.encoder.config.vocab_size
        if not forward_only_encoder:
            decoder_outputs = self.decoder(
                input_ids=dec_input_ids,
                attention_mask=dec_input_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=enc_frame_attention_mask,
                inputs_embeds=decoder_inputs_embeds,
                labels=dec_labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                use_cache=use_cache,
                past_key_values=past_key_values,
                return_dict=return_dict,
                decoder_teacher_logits=decoder_teacher_logits,
                pseudo_rows=pseudo_rows,
                decoder_entropy_loss=decoder_entropy_loss,
                vocab_size=vocab_size,
                **kwargs_decoder,
            )
            if not return_dict:
                return decoder_outputs + encoder_outputs

            return Seq2SeqLMOutput(
                loss=(encoder_outputs.loss, decoder_outputs.loss),
                logits=(encoder_outputs.logits, decoder_outputs.logits),
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_hidden_states,
                encoder_attentions=encoder_outputs.attentions,
            )
        return encoder_hidden_states.detach(), enc_frame_attention_mask.detach(), encoder_logits.detach()


class Wav2Vec2ForPreTraining_mfcc_auxiliary(Wav2Vec2ForPreTraining):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__(config)
        # 不要设置在此处，会被计入参数，并且影响多卡gpu计算（参数同步）
        # self.compute_mfcc = features.MFCC(win_length=25, hop_length=20, n_mels=40, n_mfcc=13, deltas=True,
                                          # left_frames=0, right_frames=0)
        self.mfcc_dim = 39
        
        self.mfcc_aux_weight = config.mfcc_aux_weight
        self.use_consine_loss = config.use_consine_loss
        self.mfcc_aux_z = config.mfcc_aux_z
        self.mfcc_aux_q = config.mfcc_aux_q
        self.mfcc_aux_c = config.mfcc_aux_c
        self.pred_mfcc_onlymasked = config.pred_mfcc_onlymasked
        if self.mfcc_aux_q:
            self.mfcc_projection_q = nn.Linear(self.config.proj_codevector_dim, self.mfcc_dim)
        elif self.mfcc_aux_c:
            self.mfcc_projection_c = nn.Linear(self.config.hidden_size, self.mfcc_dim)
        elif self.mfcc_aux_z:
            self.mfcc_projection_z = nn.Linear(self.config.conv_dim[-1], self.mfcc_dim)
        else:
            raise ValueError("NO AUX SELECTION")
        self.init_weights()
        

    def forward(
            self,
            input_values,
            attention_mask=None,
            mask_time_indices=None,
            sampled_negative_indices=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        mask_time_indices (:obj:`torch.BoolTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Indices to mask extracted features for contrastive loss. When in training mode, model learns to predict
            masked extracted features in `config.proj_codevector_dim` space.
        sampled_negative_indices (:obj:`torch.BoolTensor` of shape :obj:`(batch_size, sequence_length, num_negatives)`, `optional`):
            Indices indicating which quantized target vectors are used as negative sampled vectors in contrastive loss.
            Required input for pre-training.

        Returns:

        Example::

            >>> import torch
            >>> from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForPreTraining
            >>> from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices
            >>> from datasets import load_dataset
            >>> import soundfile as sf

            >>> feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("patrickvonplaten/wav2vec2-base")
            >>> model = Wav2Vec2ForPreTraining.from_pretrained("patrickvonplaten/wav2vec2-base")


            >>> def map_to_array(batch):
            ...     speech, _ = sf.read(batch["file"])
            ...     batch["speech"] = speech
            ...     return batch


            >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
            >>> ds = ds.map(map_to_array)

            >>> input_values = feature_extractor(ds["speech"][0], return_tensors="pt").input_values  # Batch size 1

            >>> # compute masked indices
            >>> batch_size, raw_sequence_length = input_values.shape
            >>> sequence_length = model._get_feat_extract_output_lengths(raw_sequence_length)
            >>> mask_time_indices = _compute_mask_indices((batch_size, sequence_length), mask_prob=0.2, mask_length=2, device=model.device)

            >>> with torch.no_grad():
            ...     outputs = model(input_values, mask_time_indices=mask_time_indices)

            >>> # compute cosine similarity between predicted (=projected_states) and target (=projected_quantized_states)
            >>> cosine_sim = torch.cosine_similarity(
            ...     outputs.projected_states, outputs.projected_quantized_states, dim=-1
            ... )

            >>> # show that cosine similarity is much higher than random
            >>> assert cosine_sim[mask_time_indices].mean() > 0.5

            >>> # for contrastive loss training model should be put into train mode
            >>> model.train()
            >>> loss = model(input_values, mask_time_indices=mask_time_indices).loss
        """
        compute_mfcc = features.MFCC(win_length=25, hop_length=20, n_mels=40, n_mfcc=13, deltas=True,
                                          left_frames=0, right_frames=0)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if mask_time_indices is not None:
            mask_time_indices = mask_time_indices.to(torch.bool)

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            mask_time_indices=mask_time_indices,
            return_dict=return_dict,
        )

        # 1. project all transformed features (including masked) to final vq dim
        transformer_features = self.project_hid(outputs[0])

        # 2. quantize all (unmasked) extracted features and project to final vq dim
        extract_features = self.dropout_features(outputs[1])
        if attention_mask is not None:
            # compute reduced attention_mask correponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)

        quantized_features, codevector_perplexity = self.quantizer(
            extract_features, mask_time_indices=mask_time_indices
        )
        quantized_features = self.project_q(quantized_features)
        
        if self.mfcc_aux_q:
            mfcc_pred = self.mfcc_projection_q(quantized_features)
        elif self.mfcc_aux_c:
            mfcc_pred = self.mfcc_projection_c(outputs[0])
        elif self.mfcc_aux_z:
            mfcc_pred = self.mfcc_projection_z(outputs[1])
        maxframe = mfcc_pred.shape[1]
        # self.compute_mfcc.compute_deltas.kernel = self.compute_mfcc.compute_deltas.kernel.cpu()
        mfcc = compute_mfcc(input_values.cpu()).to(input_values.device)
        mfcc = mfcc[:, :maxframe, :].contiguous()

        if self.pred_mfcc_onlymasked:
            mfcc_target = mfcc[mask_time_indices].reshape(-1, self.mfcc_dim)
            mfcc_pred = mfcc_pred[mask_time_indices].reshape(-1, self.mfcc_dim)
        else:
            mfcc_target = mfcc[attention_mask].reshape(-1, self.mfcc_dim)
            mfcc_pred = mfcc_pred[attention_mask].reshape(-1, self.mfcc_dim)
        assert mfcc_target.shape == mfcc_pred.shape
        if self.use_consine_loss:
            # 取值在(-1,1)间，需要归到(0,1)间
            cos_loss = F.cosine_similarity(mfcc_target, mfcc_pred)
            cos_ones = torch.ones_like(cos_loss)
            auxiliary_loss = ((cos_ones - cos_loss) * 0.5).sum()
        else:
            auxiliary_loss = F.pairwise_distance(mfcc_target,mfcc_pred,p=2).sum()
                
        loss = contrastive_loss = diversity_loss = None
        if sampled_negative_indices is not None:
            batch_size, sequence_length, hidden_size = quantized_features.shape

            # for training, we sample negatives
            # 3. sample K negatives (distractors) quantized states for contrastive loss
            # if attention_mask is passed, make sure that padded feature vectors cannot be sampled
            # sample negative quantized vectors BTC => (BxT)C
            negative_quantized_features = quantized_features.view(-1, hidden_size)[
                sampled_negative_indices.long().view(-1)
            ]
            negative_quantized_features = negative_quantized_features.view(
                batch_size, sequence_length, -1, hidden_size
            ).permute(2, 0, 1, 3)

            # 4. compute logits, corresponding to `logs = sim(c_t, [q_t, \sim{q}_t]) / \kappa`
            # of equation (3) in https://arxiv.org/pdf/2006.11477.pdf
            logits = self.compute_contrastive_logits(
                quantized_features[None, :],
                negative_quantized_features,
                transformer_features,
                self.config.contrastive_logits_temperature,
            )

            # 5. if a negative vector is identical to the positive (i.e. when codebook utilization is low),
            # its cosine similarity will be masked
            neg_is_pos = (quantized_features == negative_quantized_features).all(-1)

            if neg_is_pos.any():
                logits[1:][neg_is_pos] = float("-inf")

            # 6. compute contrastive loss \mathbf{L}_m = cross_entropy(logs) =
            # -log(exp(sim(c_t, q_t)/\kappa) / \sum_{\sim{q}} exp(sim(c_t, \sim{q})/\kappa))
            logits = logits.transpose(0, 2).reshape(-1, logits.size(0))
            target = ((1 - mask_time_indices.long()) * -100).transpose(0, 1).flatten()

            contrastive_loss = nn.functional.cross_entropy(logits.float(), target, reduction="sum")
            # 7. compute diversity loss: \mathbf{L}_d
            num_codevectors = self.config.num_codevectors_per_group * self.config.num_codevector_groups
            diversity_loss = ((num_codevectors - codevector_perplexity) / num_codevectors) * mask_time_indices.sum()

            # 8. \mathbf{L} = \mathbf{L}_m + \alpha * \mathbf{L}_d

            loss = contrastive_loss + self.config.diversity_loss_weight * diversity_loss + self.mfcc_aux_weight * auxiliary_loss

        if not return_dict:
            if loss is not None:
                return (loss, transformer_features, quantized_features, codevector_perplexity) + outputs[2:]
            return (transformer_features, quantized_features, codevector_perplexity) + outputs[2:]

        return Wav2Vec2ForPreTrainingOutput_mfcc_auxiliary(
            loss=loss,
            projected_states=transformer_features,
            projected_quantized_states=quantized_features,
            mfcc_pred=mfcc_pred,
            mfcc_target=mfcc_target,
            codevector_perplexity=codevector_perplexity,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            contrastive_loss=contrastive_loss,
            diversity_loss=diversity_loss,
            auxiliary_loss=auxiliary_loss
        )
class Wav2Vec2ForPreTraining_mfcc(Wav2Vec2PreTrainedModel):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout_features = nn.Dropout(config.feat_quantizer_dropout)
        # self.compute_mfcc = features.MFCC(win_length=25, hop_length=20, n_mels=40, n_mfcc=13, deltas=True,
                                          # left_frames=0, right_frames=0)
        # self.mfcc_norm = nn.LayerNorm(39, eps=config.layer_norm_eps)
        self.mfcc_dim = 39
        self.mfcc_projection = nn.Linear(config.hidden_size, self.mfcc_dim)
        self.pred_mfcc_onlymasked = config.pred_mfcc_onlymasked
        self.use_consine_loss = config.use_consine_loss
        self.init_weights()

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature extractor so that its parameter
        will not be updated during training.
        """
        self.wav2vec2.feature_extractor._freeze_parameters()

    def forward(
            self,
            input_values,
            attention_mask=None,
            mask_time_indices=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        compute_mfcc = features.MFCC(win_length=25, hop_length=20, n_mels=40, n_mfcc=13, deltas=True,
                                          left_frames=0, right_frames=0)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if mask_time_indices is not None:
            mask_time_indices = mask_time_indices.to(torch.bool)

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            mask_time_indices=mask_time_indices,
            return_dict=return_dict,
        )
        hidden_states, extract_features = outputs[:2]
        extract_features = self.dropout_features(extract_features)
        mfcc_pred = self.mfcc_projection(hidden_states)
        maxframe = mfcc_pred.shape[1]
        # self.compute_mfcc.compute_deltas.kernel = self.compute_mfcc.compute_deltas.kernel.cpu()
        # mfcc = self.compute_mfcc(input_values.cpu()).to(input_values.device)
        
        mfcc = compute_mfcc(input_values.cpu()).to(input_values.device)
        mfcc = mfcc[:, :maxframe, :].contiguous()
        if attention_mask is not None:
            # compute reduced attention_mask correponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)
        # print(self.pred_mfcc_onlymasked)
        # print(attention_mask.sum(dim=-1))
        if self.pred_mfcc_onlymasked:
            mfcc_target = mfcc[mask_time_indices].reshape(-1, self.mfcc_dim)
            mfcc_pred = mfcc_pred[mask_time_indices].reshape(-1, self.mfcc_dim)
        else:
            mfcc_target = mfcc[attention_mask].reshape(-1, self.mfcc_dim)
            mfcc_pred = mfcc_pred[attention_mask].reshape(-1, self.mfcc_dim)
        assert mfcc_target.shape == mfcc_pred.shape
        # print(mfcc_target.shape)
        # 取值在(-1,1)间，需要归到(0,1)间
        if self.use_consine_loss:
            cos_loss = F.cosine_similarity(mfcc_target, mfcc_pred)
            cos_ones = torch.ones_like(cos_loss)
            loss = ((cos_ones - cos_loss) * 0.5).sum()
        else:
            loss = F.pairwise_distance(mfcc_target,mfcc_pred,p=2).sum()
        if not return_dict:
            if loss is not None:
                return (loss, mfcc_pred, mfcc_target) + outputs[2:]
            return (mfcc_pred, mfcc_target) + outputs[2:]

        return Wav2Vec2ForPreTrainingOutput_mfcc(
            loss=loss,
            mfcc_pred=mfcc_pred,
            mfcc_target=mfcc_target,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,

        )
class Wav2Vec2Model_mae(Wav2Vec2Model):
    def remove_mae_mask(
        self,
        hidden_states,
        mask_time_indices,
        attention_mask,
    ):
        # 将mask取出，形成新的hidden_states以及attention_mask
        # mask_time_indices中true代表mask帧，false代表正常帧
        # attention_mask中true代表正常帧
        # print(mask_time_indices.sum(-1))
        # print(attention_mask.sum(-1))
        # print((~mask_time_indices * attention_mask).sum(-1))
        hidden_states_list = []
        attention_mask_list = []
        for index in range(hidden_states.shape[0]):
            mask_att_time_len = (~mask_time_indices * attention_mask).sum(-1)
            hidden_states_list.append(hidden_states[index][~mask_time_indices[index] * attention_mask[index]])
            attention_mask_list.append(torch.ones(mask_att_time_len[index],dtype=torch.bool))
        hidden_states = pad_sequence(hidden_states_list,True, 0)
        attention_mask = pad_sequence(attention_mask_list,True, 0).to(attention_mask.device)
        # print("!",attention_mask.sum(-1))
        return hidden_states,attention_mask
    def forward(
        self,
        input_values,
        attention_mask=None,
        mask_time_indices=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)

        hidden_states, extract_features = self.feature_projection(extract_features)
        # hidden_states = self._mask_hidden_states(
            # hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        # )
        # print(attention_mask.sum(-1))
        ##########################
        hidden_states, attention_mask = self.remove_mae_mask(hidden_states, mask_time_indices, attention_mask)
        # print(hidden_states.shape)
        # print(hidden_states)
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]
        # print(hidden_states.shape)
        # print(hidden_states)
        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]

        return Wav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
class Wav2Vec2EncoderStableLayerNorm_mae(Wav2Vec2EncoderStableLayerNorm):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.pos_conv_embed = Wav2Vec2PositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList(
            [Wav2Vec2EncoderLayerStableLayerNorm(config) for _ in range(2)]
        )
        self.gradient_checkpointing = False
class Wav2Vec2ForPreTraining_mae(Wav2Vec2PreTrainedModel):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model_mae(config)
        self.dropout_features = nn.Dropout(config.feat_quantizer_dropout)
        # self.compute_mfcc = features.MFCC(win_length=25, hop_length=20, n_mels=40, n_mfcc=13, deltas=True,
                                          # left_frames=0, right_frames=0)
        # self.mfcc_norm = nn.LayerNorm(39, eps=config.layer_norm_eps)
        
        self.mfcc_mae = Wav2Vec2EncoderStableLayerNorm_mae(config)
        self.pred_raw = False
        self.mfcc_dim = 80 if not self.pred_raw else 400
        self.mfcc_projection = nn.Linear(config.hidden_size, self.mfcc_dim)
        # self.mfcc_layer_norm = nn.LayerNorm(self.mfcc_dim, eps=config.layer_norm_eps)
        
        self.pred_mfcc_onlymasked = config.pred_mfcc_onlymasked
        self.use_consine_loss = config.use_consine_loss
        self.init_weights()

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature extractor so that its parameter
        will not be updated during training.
        """
        self.wav2vec2.feature_extractor._freeze_parameters()

    def forward(
            self,
            input_values,
            attention_mask=None,
            mask_time_indices=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        # compute_mfcc = features.MFCC(win_length=25, hop_length=20, n_mels=40, n_mfcc=13, deltas=True,
                                          # left_frames=0, right_frames=0)
        compute_mfcc = features.Fbank(win_length=25, hop_length=20, n_mels=80,left_frames=0, right_frames=0, context=False, deltas=False)
        # compute_mfcc = STFT(
                # sample_rate=16000, win_length=25, hop_length=20, n_fft=512
            # )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if mask_time_indices is not None:
            mask_time_indices = mask_time_indices.to(torch.bool)

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            mask_time_indices=mask_time_indices,
            return_dict=return_dict,
        )
        hidden_states, extract_features = outputs[:2]
        maxframe = extract_features.shape[1]
        if attention_mask is not None:
            # compute reduced attention_mask correponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)
        # 再获取一次取出mask帧后的新的_attention_mask
        _, _attention_mask = self.wav2vec2.remove_mae_mask(extract_features, mask_time_indices, attention_mask)
        hidden_states_flat = hidden_states[_attention_mask]
        # print(hidden_states_flat.shape)
        # print(extract_features.shape)
        extract_features = self.dropout_features(extract_features)
        # self.compute_mfcc.compute_deltas.kernel = self.compute_mfcc.compute_deltas.kernel.cpu()
        # mfcc = self.compute_mfcc(input_values.cpu()).to(input_values.device)
        
        if self.pred_raw:
            c = torch.nn.Conv1d(1, 400, kernel_size=(400,), stride=(320,), bias=False)
            # 构建固定卷积核, 需要大写Tensor
            # 输入(4,720)先转为(4,1,720) 经400个卷积核即(400,1,400) 得到(4,400,2)再转为(4,2,400)
            # 这400代表的是原采样点，类似图像领域的分patch
            c.weight.data = torch.Tensor(torch.eye(400).unsqueeze(1))
            c.weight.requires_grad=False
            mfcc = c(input_values.unsqueeze(1).cpu()).transpose(-2, -1).to(hidden_states_flat.device)
            # print(mfcc.shape)
        else:
            mfcc = compute_mfcc(input_values.cpu()).to(hidden_states_flat.device)
            if self.mfcc_dim == 257:
                mfcc = spectral_magnitude(mfcc)
            mfcc = mfcc[:, :maxframe, :].contiguous()
            # mfcc = self.mfcc_layer_norm(mfcc)
            # print(mfcc)
            mean = mfcc.mean(dim=-1, keepdim=True)
            var = mfcc.var(dim=-1, keepdim=True)
            mfcc = (mfcc - mean) / (var + 1.e-6)**.5
            # print(mfcc)
            
        # mfcc是未取出mask的原大小
        mae_decoder_input = torch.zeros(mfcc.shape[0],mfcc.shape[1],hidden_states.shape[2]).to(mfcc.device)
        # print(mae_decoder_input.shape)
        # mae_decoder_input = mae_decoder_input[attention_mask]
        # print(mask_time_indices.shape,mask_time_indices.sum(-1))
        # print(attention_mask.shape,attention_mask.sum(-1))
        # print(mask_time_indices.sum(-1))
        # print(attention_mask.sum(-1))
        # print((~mask_time_indices * attention_mask).sum(-1))
        mae_decoder_input[~mask_time_indices * attention_mask] = hidden_states_flat
        mae_decoder_input[mask_time_indices] = self.wav2vec2.masked_spec_embed
        # print(self.wav2vec2.masked_spec_embed)
        # print(mfcc_pred)
        
        # if attention_mask is not None:
            # # extend attention_mask
            # attention_mask_ = (1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)) * -10000.0
            # attention_mask_ = attention_mask_.expand(
                # attention_mask_.shape[0], 1, attention_mask_.shape[-1], attention_mask_.shape[-1]
            # )
        mfcc_pred = self.mfcc_mae(mae_decoder_input, attention_mask=attention_mask)[0]
        mfcc_pred = self.mfcc_projection(mfcc_pred)
        # print(mfcc_pred.shape)
        
        

        # print(self.pred_mfcc_onlymasked)
        # print(attention_mask.sum(dim=-1))
        if self.pred_mfcc_onlymasked:
            mfcc_target_ = mfcc
            mfcc_target = mfcc[mask_time_indices].reshape(-1, self.mfcc_dim)
            mfcc_pred_ = mfcc_pred
            mfcc_pred = mfcc_pred[mask_time_indices].reshape(-1, self.mfcc_dim)
        else:
            mfcc_target = mfcc[attention_mask].reshape(-1, self.mfcc_dim)
            # print(mfcc_target.shape)
            mfcc_pred = mfcc_pred[attention_mask].reshape(-1, self.mfcc_dim)
        assert mfcc_target.shape == mfcc_pred.shape
        # print(mfcc_target.shape)
        # 取值在(-1,1)间，需要归到(0,1)间
        if self.use_consine_loss:
            cos_loss = F.cosine_similarity(mfcc_target, mfcc_pred)
            cos_ones = torch.ones_like(cos_loss)
            loss = ((cos_ones - cos_loss) * 0.5).sum()
        else:
            # loss = F.pairwise_distance(mfcc_target,mfcc_pred,p=2).sum()
            loss = ((mfcc_target - mfcc_pred)**2).mean(dim=-1).sum()
        if not return_dict:
            if loss is not None:
                return (loss, mfcc_pred, mfcc_target) + outputs[2:]
            return (mfcc_pred, mfcc_target) + outputs[2:]
        if self.pred_mfcc_onlymasked:
            return Wav2Vec2ForPreTrainingOutput_mfcc(
                loss=loss,
                mfcc_pred=mfcc_pred_,
                mfcc_target=mfcc_target_,
                mae_decoder_input=mae_decoder_input,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,

            )
        return Wav2Vec2ForPreTrainingOutput_mfcc(
            loss=loss,
            mfcc_pred=mfcc_pred,
            mfcc_target=mfcc_target,
            mae_decoder_input=mae_decoder_input,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,

        )

def main():
    '''
    encoder_model_path = "/data2_from_58175/huggingface/models/wav2vec2_gpt2/encoder"
    decoder_model_path = "/data2_from_58175/huggingface/models/wav2vec2_gpt2/decoder"
    encoder = Wav2Vec2ForCTC.from_pretrained(encoder_model_path)
    # decoder = AutoModelForCausalLM.from_pretrained(decoder_model_path)
    decoder = GPT2_Decoder.from_pretrained(decoder_model_path)
    model = Wav2vec2_Gpt2(encoder=encoder, decoder=decoder)
    print(model)
    '''
    

    processor = Wav2Vec2Processor.from_pretrained("/data2_from_58175/huggingface/models/processor_aishell")
    file_path_list = load_from_disk("/home/work/w2v2_pretrain/hf_datasets_aishell2/dev")[:4]["file"]
    input_features = []
    for file_path in file_path_list:
        waveform = sox_effects.apply_effects_file(path=file_path, effects=[['rate', str(16000)]])[0]
        waveform = waveform[0]
        feature_normalized = (waveform - torch.mean(waveform)) / torch.sqrt(torch.var(waveform) + 1e-5)
        input_features.append({"input_values": feature_normalized})
    batch = processor.pad(
        input_features,
        padding=True,
        return_tensors="pt",
    )
    # model_pred_mfcc =  Wav2Vec2ForPreTraining_mfcc.from_pretrained("/data2_from_58175/huggingface/models/wav2vec2_gpt2/encoder")
    # model_pred_mfcc = Wav2Vec2ForPreTraining_mfcc.from_pretrained(
        # "/data2_from_58175/huggingface/models/wav2vec2-base_")
    config = Wav2Vec2Config.from_pretrained(
        "/data2_from_58175/huggingface/models/wav2vec2-base_",
        gradient_checkpointing=False,
    )
    config.mask_time_prob = 0.65
    model_pred_mfcc = Wav2Vec2ForPreTraining_mae(config)
    # model_pred_mfcc = Wav2Vec2ForPreTraining_mae.from_pretrained(
        # "/data2_from_58175/huggingface/models/wav2vec2-base_")
    print(model_pred_mfcc)
    # print(batch)

    device = batch["input_values"].device
    batch_size = batch["input_values"].shape[0]
    mask_indices_seq_length = model_pred_mfcc._get_feat_extract_output_lengths(
        batch["input_values"].shape[-1])
    features_shape = (batch_size, mask_indices_seq_length)
    if batch["attention_mask"] is not None:
        # compute real output lengths according to convolution formula
        batch["sub_attention_mask"] = model_pred_mfcc._get_feature_vector_attention_mask(
            mask_indices_seq_length, batch["attention_mask"]
        )
    # sample randomly masked indices
    mask_time_indices = _compute_mask_indices(
        features_shape,
        model_pred_mfcc.config.mask_time_prob,
        model_pred_mfcc.config.mask_time_length,
        attention_mask=batch['sub_attention_mask'],
    )
    batch["mask_time_indices"] = torch.tensor(mask_time_indices, dtype=torch.long, device=device)
    print(batch["mask_time_indices"].sum(-1))
    batch.pop("sub_attention_mask")
    output = model_pred_mfcc(**batch)
    print(output)


    
    '''
    model_pred_mfcc_auxiliary = Wav2Vec2ForPreTraining_mfcc_auxiliary.from_pretrained(
        "/data2_from_58175/huggingface/models/wav2vec2_gpt2/encoder")
    print(model_pred_mfcc_auxiliary)
    device = batch["input_values"].device
    batch_size = batch["input_values"].shape[0]
    mask_indices_seq_length = model_pred_mfcc_auxiliary._get_feat_extract_output_lengths(
        batch["input_values"].shape[-1])

    # make sure that no loss is computed on padded inputs
    if batch["attention_mask"] is not None:
        # compute real output lengths according to convolution formula
        batch["sub_attention_mask"] = model_pred_mfcc_auxiliary._get_feature_vector_attention_mask(
            mask_indices_seq_length, batch["attention_mask"]
        )
    features_shape = (batch_size, mask_indices_seq_length)

    # sample randomly masked indices
    mask_time_indices = _compute_mask_indices(
        features_shape,
        model_pred_mfcc_auxiliary.config.mask_time_prob,
        model_pred_mfcc_auxiliary.config.mask_time_length,
        attention_mask=batch['sub_attention_mask'],
    )
    # sample negative indices
    # 据说从所用帧中采样负样本效果更好，原文中是从mask的帧中采样
    sampled_negative_indices = _sample_negative_indices(
        features_shape,
        model_pred_mfcc_auxiliary.config.num_negatives,
        mask_time_indices=batch["sub_attention_mask"],
    )
    batch["mask_time_indices"] = torch.tensor(mask_time_indices, dtype=torch.long, device=device)
    batch["sampled_negative_indices"] = torch.tensor(sampled_negative_indices, dtype=torch.long, device=device)
    batch.pop("sub_attention_mask")
    output = model_pred_mfcc_auxiliary(**batch)
    print(output)
    '''



    # batch["labels"] = torch.ones(4,dtype=torch.long)
    # config_classify = Wav2Vec2Config.from_pretrained(
    #     "/data2_from_58175/huggingface/models/wav2vec2-base/config.json"
    # )
    # model_classify = Wav2Vec2ForSequenceClassification(config_classify)
    # print(model_classify)
    # output = model_classify(**batch)
    # print(output)
    # sio.loadmat("/data2_from_58175/huggingface/JH/x_train.mat")

# 在被import时__name__不等于__main__，则不会进入main(), 当直接执行本脚本时，__name__=__main__
if __name__ == "__main__":
    main()
