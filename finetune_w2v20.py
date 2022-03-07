#!/usr/bin/env python3
import torchaudio
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, PretrainedConfig, Wav2Vec2Model
import time
from tqdm import tqdm
from typing import Optional, Tuple
import math
import glob
from scipy import signal
import collections
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    PredictionOutput,
    ShardedDDPOption,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    denumpify_detensorize,
    get_last_checkpoint,
    set_seed,
    speed_metrics,
)
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    ShardSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from transformers.file_utils import (
    CONFIG_NAME,
    WEIGHTS_NAME,
    PushToHubMixin,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    is_training_run_on_sagemaker,
)
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from torch.utils.data.dataloader import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers.modeling_utils import PreTrainedModel
import logging
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Union
import random
import torchaudio.sox_effects as sox_effects
import torchaudio as ta
import datasets
import numpy as np
import torch
import inspect
from packaging import version
from torch import nn
from lang_trans import arabic
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    is_apex_available,
    trainer_utils,
    AutoConfig,
    is_wandb_available,
)
from datasets import load_dataset, load_from_disk
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["TRANSFORMERS_CACHE"] = "/data2_from_58175/huggingface/transformers"
os.environ["HF_DATASETS_CACHE"] = "/data2_from_58175/huggingface/datasets"
os.environ["HF_METRICS_CACHE"] = "/data2_from_58175/huggingface/metrics"
os.environ["HF_HOME"] = "/data2_from_58175/huggingface"
os.environ["TMPDIR"] = "/work/tmp"

if is_apex_available():
    from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

logger = logging.getLogger(__name__)


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_random_seed(42, deterministic=False)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    processor_path: str = field(
        metadata={"help": "Path to processor"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    freeze_feature_extractor: Optional[bool] = field(
        default=True, metadata={"help": "Whether to freeze the feature extractor layers of the model."}
    )
    freeze_all_except_lm: Optional[bool] = field(
        default=False, metadata={"help": "Whether to freeze all parameters of the model except lm_head."}
    )

    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "Whether to freeze the feature extractor layers of the model."}
    )
    verbose_logging: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to log verbose messages or not."},
    )
    freeze_ALN: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to freeze freeze parameters of freeze_feature_extractor and feed_forward."}
    )
    reinit_lm_head: Optional[bool] = field(
        default=False, metadata={"help": "Whether to reinitial lm_head"}
    )
    encoder_decoder_mode: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use gpt2 as decoder"}
    )



def configure_logger(model_args: ModelArguments, training_args: TrainingArguments):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging_level = logging.WARNING
    if model_args.verbose_logging:
        logging_level = logging.DEBUG
    elif trainer_utils.is_main_process(training_args.local_rank):
        logging_level = logging.INFO
    logger.setLevel(logging_level)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: str = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_path: str = field(
        default=None, metadata={"help": "The path of the dataset(including subset)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_split_name: Optional[str] = field(
        default="train",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    validation_split_name: Optional[str] = field(
        default="validation",
        metadata={
            "help": "The name of the validation data set split to use (via the datasets library). Defaults to 'validation'"
        },
    )
    target_text_column: Optional[str] = field(
        default="text",
        metadata={"help": "Column in the dataset that contains label (target text). Defaults to 'text'"},
    )
    speech_file_column: Optional[str] = field(
        default="file",
        metadata={"help": "Column in the dataset that contains speech file path. Defaults to 'file'"},
    )
    target_feature_extractor_sampling_rate: Optional[bool] = field(
        default=False,
        metadata={"help": "Resample loaded audio to target feature extractor's sampling rate or not."},
    )
    max_duration_in_seconds: Optional[float] = field(
        default=None,
        metadata={"help": "Filters out examples longer than specified. Defaults to no filtering."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    speed_perturb: Optional[bool] = field(
        default=False,
        metadata={"help": "apply speed perpturbation in collator."},
    )

class Add_Noise_Reverb(object):

    def __init__(self, musan_path, rir_path):

        self.noisetypes = ['noise', 'speech', 'music']

        self.noisesnr = {'noise': [40, 50], 'speech': [40, 50], 'music': [40, 50]}
        # 三种类型噪声的采样数目，用于叠加多段，speech使用4-7段(模拟多人背景噪声)，noise或者music只使用一段
        self.numnoise = {'noise': [1, 1], 'speech': [3, 7], 'music': [1, 1]}
        # noiselist的keys为 'noise'、'speech'、'music'
        # values是对应的文件列表
        self.noiselist = {}
        # musan_path="/tsdata/sre/musan"
        augment_files = glob.glob(os.path.join(musan_path, '*/*/*.wav'))

        for file in augment_files:
            if not file.split('/')[-3] in self.noiselist:
                self.noiselist[file.split('/')[-3]] = []
            self.noiselist[file.split('/')[-3]].append(file)
        # simulated_rirs_files有60000条
        # real_point_rirs_files有417条
        # pointsource_noises有843条
        # 总共有61260条混响样本
        self.simulated_rirs_files = glob.glob(os.path.join(rir_path, '*/*/*/*.wav'))
        #
        self.real_point_rirs_files = glob.glob(os.path.join(rir_path, '*/*.wav'))
        #         self.rir_files = glob.glob(os.path.join("/tsdata/noise/RIRS_NOISES/real_rirs_isotropic_noises/",'*.wav'))
        self.rir_files = self.simulated_rirs_files + self.real_point_rirs_files
        print(f"noisetypes:{self.noisetypes}")
        for k, v in self.noiselist.items():
            print(f"noisetype and num_files: {k, len(v)}")
        print(f"num of rir_files:{len(self.rir_files)}")

    def additive_noise(self, noisecat, audio):
        # audio为numpy 2Darray
        # noisecat: {noise、speech、music}三选一

        clean_db = 10 * np.log10(np.mean(audio ** 2) + 1e-4)

        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))

        noises = []
        # speech选择4-7段，noise或者music只选择一段
        audio_length = audio.shape[1]
        for noise in noiselist:
            noiseaudio, sample_rate = ta.load(noise)
            noiseaudio = noiseaudio.detach().numpy()
            noise_length = noiseaudio.shape[1]
            if noise_length <= audio_length:
                shortage = audio_length - noise_length + 1
                # 若是填充一维数组 arr1D=np.array([1, 1, 2, 2, 3, 4])
                # np.pad(arr1D, (2, 3), 'wrap')代表首部补两个，尾部补三个
                # mode=warp决定用于填补的元素，warp是首尾相连型，得到[3, 4, 1, 1, 2, 2, 3, 4, 1, 1, 2]
                # 此例是二维数组，所以要补充(0,0)用于操作第一个维度，(0,0)即表示第一维度不变
                noiseaudio = np.pad(noiseaudio, ((0, 0), (0, shortage)), 'wrap')
                noiseaudio = noiseaudio[:, :audio_length]
            else:
                # 噪声样本长，则随机选与音频等长的段，不一定只取开头，这可以通过startframe的随机选择实现
                startframe = np.int64(random.random() * (noise_length - audio_length))
                noiseaudio = noiseaudio[:, int(startframe):int(startframe) + audio_length]
            noise_snr = random.uniform(self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])
            noise_db = 10 * np.log10(np.mean(noiseaudio[0] ** 2) + 1e-4)
            noises.append(np.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noiseaudio)
        noise_sum = np.sum(np.concatenate(noises, axis=0), axis=0, keepdims=True)
        return noise_sum + audio

    def reverberate(self, audio):

        rir_file = random.choice(self.rir_files)
        audio_length = audio.shape[1]
        rir, fs = ta.load(rir_file)
        rir = rir.detach().numpy()
        rir_length = rir.shape[1]
        if rir_length <= audio_length:
            shortage = rir_length - rir_length + 1
            rir = np.pad(rir, ((0, 0), (0, shortage)), 'wrap')
        else:
            startframe = np.int64(random.random() * (rir_length - audio_length))
            rir = rir[:, int(startframe):int(startframe) + audio_length]
        rir = rir / np.sqrt(np.sum(rir ** 2))

        return signal.convolve(audio, rir, mode='full')[:, :audio_length]


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = []
        if "speech" in features[0].keys():
            # print("proccess speech")
            for feature in features:
                feature_normalized = (feature["speech"] - np.mean(feature["speech"])) / np.sqrt(
                    np.var(feature["speech"]) + 1e-5)
                input_features.append({"input_values": feature_normalized})
                # input_features = [{"input_values": feature["input_values"]} for feature in features]
        elif "file" in features[0].keys():
            # 终版，问题出在input_values已经是normalize后的，而torchaudio处理原始信号比较精准，所以我们采用speech作为处理信号，而后再归一化
            for feature in features:
                # torchaudio需要二维数组(1,*)，输出也是二维数组，所以再调用时后输出后需要unsqueeze(0)和squeeze(0)
                # 法一
                # sr = torchaudio.backend.sox_io_backend.info(feature["file"]).sample_rate
                # start_frame = int(feature["seg_start"] * sr)
                # end_frame = int(feature["seg_end"] * sr)
                # waveform, _ = torchaudio.backend.sox_io_backend.load(
                # filepath=feature["file"],
                # num_frames=end_frame - start_frame,
                # frame_offset=start_frame
                # )
                # waveform = torchaudio.sox_effects.apply_effects_tensor(
                # waveform, sr,[['rate', str(16000)]])[0]

                # waveform = waveform[0]
                # feature_normalized = (waveform - torch.mean(waveform)) / torch.sqrt(torch.var(waveform) + 1e-5)
                # input_features.append({"input_values": feature_normalized})
                # 法二
                waveform = sox_effects.apply_effects_file(path=feature["file"], effects=[['rate', str(16000)]])[0]
                waveform = waveform[0]
                feature_normalized = (waveform - torch.mean(waveform)) / torch.sqrt(torch.var(waveform) + 1e-5)
                input_features.append({"input_values": feature_normalized})
        else:
            raise ValueError(f" ['file'] or ['speech'] is required for collect func")

        batch = self.processor.pad(
            input_features,
            padding=True,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        # print(batch["input_values"].dtype)
        if "labels" in features[0].keys():
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            # print(batch["input_values"].dtype)
        else:
            assert ("text" in features[0].keys()),f" ['labels'] or ['text'] is required for training"
            # print(features)
            label_features = [{"input_ids": self.processor.tokenizer(feature["text"]).input_ids} for feature in features]
        # print(label_features)
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=True,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        # print(batch["input_values"][:,-5:])
        # print(batch["attention_mask"][:,-5:])
        # print(batch["labels"][:,-5:])
        return batch


@dataclass
class DataCollatorCTCWithPadding_Speed_Perturb:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    speeds = np.linspace(0.9, 1.1, 3).tolist()
    sp_weights = (np.ones(3)).tolist()
    #     sp_weights = (np.ones(21)*(1-0.3)/20).tolist()
    #     sp_weights[10] = 0.5
    #     print(sp_weights)
    #     speeds = [0.9,1.0,1.1]
    #     sp_weights = [1,1,1]
    logger.info("doing speed perturbation in collect_func") if len(speeds) > 1 else None
    add_no_re = False
    logger.info("adding noise in collect_func") if add_no_re else None
    add_noise_reverb = Add_Noise_Reverb(musan_path="/tsdata/sre/musan",
                                        rir_path="/tsdata/sre/RIRS_NOISES/") if add_no_re else None

    # pitchs = np.linspace(-100,100,21).tolist()
    # pit_weights = np.ones(21).tolist()
    # volumns = np.linspace(0.12,2,81).tolist()
    # vol_weights = np.ones(81).tolist()
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods

        # print([feature["labels"][:1] for feature in features])
        speed = random.choices(self.speeds, self.sp_weights, k=1)[0]
        # sr = 16000
        # volumn = random.choices(self.volumns, self.vol_weights, k=1)[0]
        # pitch = random.choices(self.pitchs, self.pit_weights, k=1)[0]
        input_features = []
        if "speech" in features[0].keys():
            for feature in features:
                if speed != 1.0:
                    # torchaudio需要二维数组(1,*)，输出也是二维数组，所以再调用时后输出后需要unsqueeze(0)和squeeze(0)
                    feature_speed_perturbed = sox_effects.apply_effects_tensor(
                        tensor=torch.tensor(feature["speech"]).unsqueeze(0),
                        sample_rate=16000,
                        effects=[['speed', str(speed)], ['rate', str(16000)]]
                    )[0]
                else:
                    feature_speed_perturbed = torch.tensor(feature["speech"]).unsqueeze(0)
                if self.add_no_re and speed == 1.0:
                    # 1-4则加混响或者加噪
                    augtype = random.choices([2, 3, 4], [1, 1, 1], k=1)[0]
                    feature_speed_perturbed_np = feature_speed_perturbed.numpy()
                    # 输入为2D array
                    if augtype == 1:
                        audio_aug = self.add_noise_reverb.reverberate(feature_speed_perturbed_np)[0]
                    elif augtype == 2:
                        audio_aug = self.add_noise_reverb.additive_noise('music', feature_speed_perturbed_np)[0]
                    elif augtype == 3:
                        audio_aug = self.add_noise_reverb.additive_noise('speech', feature_speed_perturbed_np)[0]
                    elif augtype == 4:
                        audio_aug = self.add_noise_reverb.additive_noise('noise', feature_speed_perturbed_np)[0]
                    else:
                        audio_aug = feature_speed_perturbed_np[0]
                        # audio_aug : 1D array
                    audio_aug = (audio_aug - np.mean(audio_aug)) / np.sqrt(np.var(audio_aug) + 1e-5)
                else:
                    feature_speed_perturbed = feature_speed_perturbed[0]
                    audio_aug = (feature_speed_perturbed - torch.mean(feature_speed_perturbed)) / torch.sqrt(
                        torch.var(feature_speed_perturbed) + 1e-5)
                input_features.append({"input_values": audio_aug})
        elif "file" in features[0].keys():
            # 无法直接使用apply_effects_file，因为还需要根据seg_start和seg_end对segments音频进行裁出
            for feature in features:
                # torchaudio需要二维数组(1,*)，输出也是二维数组，所以再调用时后输出后需要unsqueeze(0)和squeeze(0)
                # logger.info("load opus")
                # 法一：根据file seg_start seg_end切出样本，在进行在线扩增降采样，但有的oups太大，速度很麻
                # sr = torchaudio.backend.sox_io_backend.info(feature["file"]).sample_rate
                # start_frame = int(feature["seg_start"] * sr)
                # end_frame = int(feature["seg_end"] * sr)
                # waveform, _ = torchaudio.backend.sox_io_backend.load(
                # filepath=feature["file"],
                # num_frames=end_frame - start_frame,
                # frame_offset=start_frame
                # )
                # feature_speed_perturbed = torchaudio.sox_effects.apply_effects_tensor(
                # waveform, sr,[['speed', str(speed)], ['rate', str(16000)]])[0]
                # 法二：事先离线将样本切割出并存储，此使batch["file"]以更新为样本对应得flac文件
                if speed != 1.0:
                    feature_speed_perturbed = sox_effects.apply_effects_file(path=feature["file"],
                                                                             effects=[['speed', str(speed)],
                                                                                      ['rate', str(16000)]])[0]
                else:
                    feature_speed_perturbed = sox_effects.apply_effects_file(path=feature["file"], effects=[['rate', str(16000)]])[0]
                if self.add_no_re and speed == 1.0:
                    # 1-4则加混响或者加噪
                    augtype = random.choices([2, 3, 4], [1, 1, 1], k=1)[0]
                    feature_speed_perturbed_np = feature_speed_perturbed.numpy()
                    # 输入为2D array
                    if augtype == 1:
                        audio_aug = self.add_noise_reverb.reverberate(feature_speed_perturbed_np)[0]
                    elif augtype == 2:
                        audio_aug = self.add_noise_reverb.additive_noise('music', feature_speed_perturbed_np)[0]
                    elif augtype == 3:
                        audio_aug = self.add_noise_reverb.additive_noise('speech', feature_speed_perturbed_np)[0]
                    elif augtype == 4:
                        audio_aug = self.add_noise_reverb.additive_noise('noise', feature_speed_perturbed_np)[0]
                    else:
                        audio_aug = feature_speed_perturbed_np[0]
                        # audio_aug : 1D array
                    audio_aug = (audio_aug - np.mean(audio_aug)) / np.sqrt(np.var(audio_aug) + 1e-5)
                else:
                    feature_speed_perturbed = feature_speed_perturbed[0]
                    audio_aug = (feature_speed_perturbed - torch.mean(feature_speed_perturbed)) / torch.sqrt(
                        torch.var(feature_speed_perturbed) + 1e-5)
                input_features.append({"input_values": audio_aug})
        else:
            raise ValueError(f" ['file'] or ['speech'] is required for speech perpturbation ")
        batch = self.processor.pad(
            input_features,
            padding=True,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        if "labels" in features[0].keys():
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            # print(batch["input_values"].dtype)
        else:
            assert ("text" in features[0].keys()),f" ['labels'] or ['text'] is required for training"
            # print(features)
            label_features = [{"input_ids": self.processor.tokenizer(feature["text"]).input_ids} for feature in features]
            # print(label_features)
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=True,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        # print(batch["input_values"][:,-5:])
        # print(batch["attention_mask"][:,-5:])
        # print(batch["labels"][:,-5:])
        return batch

class CTCTrainer(Trainer):
    def __init__(
            self,
            model: Union[PreTrainedModel, nn.Module] = None,
            args: TrainingArguments = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Callable[[], PreTrainedModel] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            processor: Optional[Wav2Vec2Processor] = None,
            pseudo_model: Union[PreTrainedModel, nn.Module] = None,
            data_collator_eval: Optional[DataCollator] = None,
            pseudo_onthefly: Optional[bool] = None,
            teacher_model: Union[PreTrainedModel, nn.Module] = None,
            eval_only_encoder: Optional[bool] = None,
    ):
        super(CTCTrainer, self, ).__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
        )
        self.processor = processor
        self.pseudo_model = pseudo_model
        self.pseudo_onthefly = pseudo_onthefly
        self.data_collator_eval = data_collator_eval
        self.teacher_model = teacher_model
        self.eval_only_encoder = eval_only_encoder

    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        global input_column_change
        if not self.args.remove_unused_columns:
            return dataset
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += ["label", "label_ids"]
        columns = [k for k in self._signature_columns if k in dataset.column_names]
        ignored_columns = list(set(dataset.column_names) - set(self._signature_columns))

        # 同一保留speech，可用于做变速，并重新做normalization，即为input_values
        ignored_columns.remove("speech") if "speech" in ignored_columns else None
        ignored_columns.remove("file") if "file" in ignored_columns else None
        ignored_columns.remove("text") if "text" in ignored_columns else None
        # ignored_columns.remove("seg_end") if "seg_end" in ignored_columns else None
        # ignored_columns.remove("seg_start") if "seg_start" in ignored_columns else None

        # if input_column_change:
        # ignored_columns.remove("speech")
        # ignored_columns.append("input_values")
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set "
            logger.info(
                f"The following columns {dset_description} don't have a corresponding argument in "
                f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
            )

        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        # if flag:
        # torch.save(inputs, "./input_values_.pt")
        # flag = False
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                kwargs = dict(device=self.args.device)
                if self.deepspeed and inputs[k].dtype != torch.int64:
                    # NLP models inputs are int64 and those get adjusted to the right dtype of the
                    # embedding. Other models such as wav2vec2's inputs are already float and thus
                    # may need special handling to match the dtypes of the model
                    kwargs.update(dict(dtype=self.args.hf_deepspeed_config.dtype()))

                inputs[k] = v.to(**kwargs)

        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        return inputs

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation :class:`~torch.utils.data.DataLoader`.

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
                If provided, will override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`, columns not
                accepted by the ``model.forward()`` method are automatically removed. It must implement :obj:`__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")

        if isinstance(eval_dataset, torch.utils.data.dataset.IterableDataset):
            if self.args.world_size > 1:
                eval_dataset = IterableDatasetShard(
                    eval_dataset,
                    batch_size=self.args.eval_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )
            return DataLoader(
                eval_dataset,
                batch_size=self.args.eval_batch_size,
                collate_fn=self.data_collator_eval,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        eval_sampler = self._get_eval_sampler(eval_dataset)
        print("~" * 20, "using data_collator_eval with no augmentation", "~" * 20)
        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator_eval,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def evaluation_loop(
            self,
            dataloader: DataLoader,
            description: str,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

        Works both with or without labels.
        """
        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        # if eval is called w/o train init deepspeed here
        if self.args.deepspeed and not self.deepspeed:
            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(self, num_training_steps=0, resume_from_checkpoint=None)
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            # XXX: we don't need optim/sched for inference, but this needs to be sorted out, since
            # for example the Z3-optimizer is a must for zero3 to work even for inference - what we
            # don't need is the deepspeed basic optimizer which is self.optimizer.optimizer
            deepspeed_engine.optimizer.optimizer = None
            deepspeed_engine.lr_scheduler = None

        model = self._wrap_model(self.model, training=False)

        # if full fp16 is wanted on eval and this ``evaluation`` or ``predict`` isn't called while
        # ``train`` is running, halve it first and then put on device
        if not self.is_in_train and self.args.fp16_full_eval:
            model = model.half().to(self.args.device)

        batch_size = dataloader.batch_size

        logger.info(f"***** Running {description} *****")
        if isinstance(dataloader.dataset, collections.abc.Sized):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

        if self.args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size

            # Prediction step
            if self.eval_only_encoder:
                loss, logits, labels = self.prediction_step(model.encoder, inputs, prediction_loss_only,
                                                            ignore_keys=ignore_keys)
            else:
                loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only,
                                                            ignore_keys=ignore_keys)

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if self.args.eval_accumulation_steps is not None and (step + 1) % self.args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if not isinstance(eval_dataset, IterableDataset):
            num_samples = len(eval_dataset)
        elif isinstance(eval_dataset, IterableDatasetShard):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, loss_tuple):
        if self.control.should_log:
            logs: Dict[str, float] = {}
            tr_loss_scalar = tr_loss.item()
            ctc_loss, additional_loss = loss_tuple[0].detach(), loss_tuple[1].detach()
            ctc_loss_scalar = ctc_loss.item()
            additional_loss_scalar = additional_loss.item()
            # reset tr_loss to zero
            tr_loss -= tr_loss
            ctc_loss -= ctc_loss
            additional_loss -= additional_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            #########################################################
            additional_loss
            if additional_loss_scalar == 0:
                # 处于纯ctc模式
                logs["ctc_loss"] = logs["loss"]
            else:
                logs["ctc_loss"] = round(ctc_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["additional_loss"] = round(
                additional_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._total_ctc_loss_scalar += ctc_loss_scalar
            self._total_additional_loss_scalar += additional_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)
            if is_wandb_available():
                wandb.log(logs)
        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate()
            self._report_to_hp_search(trial, epoch, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def train(
            self,
            resume_from_checkpoint: Optional[Union[str, bool]] = None,
            trial: Union["optuna.Trial", Dict[str, Any]] = None,
            **kwargs,
    ):
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (:obj:`str` or :obj:`bool`, `optional`):
                If a :obj:`str`, local path to a saved checkpoint as saved by a previous instance of
                :class:`~transformers.Trainer`. If a :obj:`bool` and equals `True`, load the last checkpoint in
                `args.output_dir` as saved by a previous instance of :class:`~transformers.Trainer`. If present,
                training will resume from the model/optimizer/scheduler states loaded here.
            trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            kwargs:
                Additional keyword arguments used to hide deprecated arguments
        """

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        args = self.args

        self.is_in_train = True

        # do_train is not a reliable argument, as it might not be set and .train() still called, so
        # the following is a workaround:
        if args.fp16_full_eval and not args.do_train:
            self.model = self.model.to(args.device)

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )
        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)

        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            set_seed(args.seed)
            self.model = self.call_model_init(trial)
            model_reloaded = True
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")

        if resume_from_checkpoint is not None:
            if not os.path.isfile(os.path.join(resume_from_checkpoint, WEIGHTS_NAME)):
                raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

            logger.info(f"Loading model from {resume_from_checkpoint}).")

            if os.path.isfile(os.path.join(resume_from_checkpoint, CONFIG_NAME)):
                config = PretrainedConfig.from_json_file(os.path.join(resume_from_checkpoint, CONFIG_NAME))
                checkpoint_version = config.transformers_version
                if checkpoint_version is not None and checkpoint_version != __version__:
                    logger.warn(
                        f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
                        f"Transformers but your current version is {__version__}. This is not recommended and could "
                        "yield to errors or unwanted behaviors."
                    )

            if args.deepspeed:
                # will be resumed in deepspeed_init
                pass
            else:
                # We load the model state dict on the CPU to avoid an OOM error.
                state_dict = torch.load(os.path.join(resume_from_checkpoint, WEIGHTS_NAME), map_location="cpu")
                # If the model is on the GPU, it still works!
                self._load_state_dict_in_model(state_dict)

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self.model = self.model.to(args.device)
            self.model_wrapped = self.model

        # Keeping track whether we can can len() on the dataset or not
        train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size
        if train_dataset_is_sized:
            num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training datalaoder has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = len(self.train_dataset) * args.num_train_epochs
        else:
            # see __init__. max_steps is set when the dataset has no __len__
            max_steps = args.max_steps
            num_train_epochs = int(args.num_train_epochs)
            num_update_steps_per_epoch = max_steps
            num_train_samples = args.max_steps * total_train_batch_size

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = self.sharded_ddp is not None and self.sharded_ddp != ShardedDDPOption.SIMPLE
        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        model = self._wrap_model(self.model_wrapped)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        num_examples = (
            self.num_examples(train_dataloader) if train_dataset_is_sized else total_train_batch_size * args.max_steps
        )

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
                os.path.join(resume_from_checkpoint, "trainer_state.json")
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, "trainer_state.json"))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
        self.state.trial_params = hp_params(trial) if trial is not None else None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        #########################################################
        ctc_loss = torch.tensor(0.0).to(args.device)
        additional_loss = torch.tensor(0.0).to(args.device)
        self._total_ctc_loss_scalar = 0.0
        self._total_additional_loss_scalar = 0.0

        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                # We just need to begin an iteration to create the randomization of the sampler.
                for _ in train_dataloader:
                    break

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator) if train_dataset_is_sized else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)
            #             onthefly_point = [int(num_train_epochs*i) for i in np.arange(10)/10]
            #             if epoch in onthefly_point:
            #                 print(onthefly_point)
            #                 print(epoch)
            #                 state_dict_last_epoch = copy.deepcopy(model.cpu().state_dict())
            #                 model_last_epoch = self.pseudo_model.load_state_dict(state_dict_last_epoch)
            for step, inputs in enumerate(epoch_iterator):
                # Skip past any already trained steps if resuming training
                # print(inputs["input_values"][:,-5:])
                # print(inputs["attention_mask"][:,-5:])
                # print(inputs["labels"][:,-5:])
                #                 set_trace()
                #####################################
                # if self.state.global_step <= 11940 and self.state.global_step>=4:
                # print(f"self.state.global_step = {self.state.global_step}")
                # print(f"self.state.epoch = {self.state.epoch}")
                # self.state.global_step += 1
                # self.state.epoch = epoch + (step + 1) / steps_in_epoch
                # continue
                if self.pseudo_onthefly:
                    inputs = self._prepare_inputs(inputs)
                    with torch.no_grad():
                        pseudo_logits = model_last_epoch(inputs["input_values"].cpu()).logits
                    pseudo_ids = torch.max(pseudo_logits, dim=-1)[1]
                    pseudo_trans = self.processor.batch_decode(pseudo_ids)
                    # 个别样本是静音段但是带标签，导致trans=""，processor.as_target_processor()无法处理空字符串，可将""转为" "
                    pseudo_trans = [" " if i == "" else i for i in pseudo_trans]

                    pseudo_mask = (inputs["labels"] == -1)
                    # 得到伪标签样本在当前batch的行索引值
                    pseudo_rows = pseudo_mask.nonzero()[:, 0].tolist()

                    with self.processor.as_target_processor():
                        labels_onthefly = self.processor(pseudo_trans).input_ids
                        # print(f"labels_onthefly:{labels_onthefly}")
                    labels = inputs["labels"].tolist()
                    # 将labels中-1所在行的伪标签id换成模型此刻预测的token_id
                    for i in pseudo_rows:
                        # 仍然在伪标签前加上"-1"标志
                        labels[i] = [-1] + labels_onthefly[i]
                    # 新labels需要重新pad
                    #                     temp_labels = inputs["labels"]
                    inputs["labels"] = pad_sequence([torch.from_numpy(np.array(i)) for i in labels], batch_first=True,
                                                    padding_value=-100)
                #                     print(temp_labels==inputs["labels"])
                # print(f"onethefly labels:{labels}")
                # print(f"onethefly labels.shape:{labels.shape}")

                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                if (
                        ((step + 1) % args.gradient_accumulation_steps != 0)
                        and args.local_rank != -1
                        and args._no_sync_in_gradient_accumulation
                ):
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    #########################################################
                    with model.no_sync():
                        if not self.eval_only_encoder:
                            # set_trace()
                            tr_loss_ = self.training_step(model, inputs, teacher_model=self.teacher_model)
                            tr_loss += tr_loss_
                            loss_tuple = torch.tensor(0), torch.tensor(0)
                        else:
                            tr_loss_, loss_tuple_ = self.training_step(model, inputs, teacher_model=self.teacher_model)
                            ctc_loss_, additional_loss_ = loss_tuple_
                            # print(tr_loss,ctc_loss,additional_loss)
                            tr_loss += tr_loss_
                            ctc_loss += ctc_loss_.detach()
                            additional_loss += additional_loss_.detach()
                            loss_tuple = (ctc_loss, additional_loss)
                else:
                    #                     print(self.teacher_model)
                    #########################################################
                    if not self.eval_only_encoder:
                        # set_trace()
                        tr_loss_ = self.training_step(model, inputs, teacher_model=self.teacher_model)
                        tr_loss += tr_loss_
                        loss_tuple = torch.tensor(0), torch.tensor(0)
                    else:
                        tr_loss_, loss_tuple_ = self.training_step(model, inputs, teacher_model=self.teacher_model)
                        tr_loss += tr_loss_
                        ctc_loss_, additional_loss_ = loss_tuple_
                        ctc_loss += ctc_loss_.detach()
                        additional_loss += additional_loss_.detach()
                        loss_tuple = (ctc_loss, additional_loss)
                # print(f"tr_loss={tr_loss}")
                self.current_flos += float(self.floating_point_ops(inputs))

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        steps_in_epoch <= args.gradient_accumulation_steps
                        and (step + 1) == steps_in_epoch
                ):
                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                        # deepspeed does its own clipping

                        if self.use_amp:
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    optimizer_was_run = True
                    if self.deepspeed:
                        pass  # called outside the loop
                    elif is_torch_tpu_available():
                        xm.optimizer_step(self.optimizer)
                    elif self.use_amp:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                        # print(f"scale_before = {scale_before}")
                        # print(f"scale_after = {scale_after}")
                    else:
                        self.optimizer.step()

                    if optimizer_was_run and not self.deepspeed:
                        self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, loss_tuple)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, loss_tuple)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.local_rank != -1:
                dist.barrier()

            logger.info(
                f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            )
            # We load the model state dict on the CPU to avoid an OOM error.
            state_dict = torch.load(os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME), map_location="cpu")
            # If the model is on the GPU, it still works!
            self._load_state_dict_in_model(state_dict)

            if self.deepspeed:
                self.deepspeed.load_checkpoint(
                    self.state.best_model_checkpoint, load_optimizer_states=False, load_lr_scheduler_states=False
                )

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        # print(f"sigle gpu loss:{loss}")
        return (loss, outputs) if return_outputs else loss

    def compute_kl_loss(self, p, q):
        q_loss = F.kl_div(q.log_softmax(dim=-1), p.softmax(dim=-1), reduction='sum')
        p_loss = F.kl_div(p.log_softmax(dim=-1), q.softmax(dim=-1), reduction='sum')
        return (q_loss + p_loss) / 2

    def compute_distill_loss(self, model, teacher_model, inputs, return_outputs=False):
        """
        计算帧级别相对熵损失，例如[[[0.6,0.4],[0.3,0.7]]]，制作其标签为[[[1,0],[0,1]]]
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        # num_char_tokens = (inputs["labels"] >= 0).sum()
        # print(f"num_char_tokens={num_char_tokens}")
        outputs = model(**inputs)
        logits = outputs["logits"]
        with torch.no_grad():
            teacher_model.eval()
            teacher_outputs_logits = model.to(logits.device)(**inputs)["logits"]
        #         log_probs_mat = logits
        # print(f"log_probs_mat.shape={log_probs_mat.shape}")
        #         log_probs_mat_mask = log_probs_mat.argmax(dim=-1,keepdim=True)
        #         log_probs_mat_onehot = torch.zeros(log_probs_mat.shape,device=log_probs_mat.device).scatter_(-1, log_probs_mat_mask, 1)
        # print(f"log_probs_mat_onehot.shape={log_probs_mat_onehot.shape}")
        # print(f"log_probs_mat_mask.shape={log_probs_mat_mask.shape}")
        distill_loss = self.compute_kl_loss(logits, teacher_outputs_logits)
        # kl_loss = self.compute_kl_loss(log_probs_mat,log_probs_mat_onehot)/num_char_tokens
        #         distill_weight = 0.001

        # Save past state if it exist
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"]
            #             print(f"ctc_loss={loss}")
            #             print(f"distill_loss.requires_grad={distill_loss.requires_grad}")
            loss = (loss, distill_loss)
        #             print(f"distill_weight*distill_loss={distill_weight*distill_loss}")
        # print(f"total_loss={loss}")
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]],
                      teacher_model) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        # global flag
        model.train()
        inputs = self._prepare_inputs(inputs)
        # print(inputs["input_values"][0,-5:])
        # print(inputs["attention_mask"][0,-5:])
        # print(inputs["labels"][0,-5:])

        # if flag:
        # torch.save(inputs, "./input_values_.pt")
        # flag = False
        if self.use_amp:
            with autocast():
                #########################################################
                if not teacher_model:
                    if self.eval_only_encoder:
                        # training_step DDP 各卡算各自的，每批都会进行loss的反向传播并计算梯度
                        ctc_loss, att_loss = self.compute_loss(model, inputs)
                        # print(ctc_loss,"\n",att_loss)
                    #                         print(f"sum ctc_loss",{ctc_loss.item()} )
                    #                         print(f"att_loss={att_loss.item()}")
                    else:
                        loss = self.compute_loss(model, inputs)
                else:
                    ctc_loss, distill_loss = self.compute_distill_loss(model, teacher_model, inputs)
            # print(f"loss with autocast={loss}")
        else:
            loss = self.compute_loss(model, inputs)
            # print(f"loss without autocast={loss}")
        # 有点问题，DDP下self.args.n_gpu被设置为1，则不进入if
        # if self.args.n_gpu > 1:
        # if model.module.config.ctc_loss_reduction == "mean":
        # loss = loss.mean()
        # elif model.module.config.ctc_loss_reduction == "sum":
        # loss = loss.sum() / (inputs["labels"] >= 0).sum()
        # else:
        # raise ValueError(f"{model.config.ctc_loss_reduction} is not valid. Choose one of ['mean', 'sum']")
        ctc_weight = 0.2
        if model is not self.model:
            # print("####################model is wrapped####################")
            # print(f"####################type(model):{type(model)}####################")
            #########################################################
            if self.eval_only_encoder:
                if model.module.encoder.config.ctc_loss_reduction == "mean":
                    ctc_loss = ctc_loss.mean()
                elif model.module.encoder.config.ctc_loss_reduction == "sum":
                    # print(f"valid_num_labels={(inputs['labels'] >= 0).sum()}")
                    ctc_loss = ctc_loss.sum() / inputs["labels"].shape[0]
                    # cross_entropy loss默认就是batch mean
                    att_loss = att_loss.sum() / inputs["labels"].shape[0]
                    #                     print(f"mean ctc_loss={loss}")
                    if not torch.isnan(att_loss):
                        # 正常情况下att_loss不为nan
                        loss = ctc_weight * ctc_loss + (1 - ctc_weight) * att_loss
                    else:
                        print("!" * 20, "att_loss is nan , let loss = ctc_loss, att_loss=0", "!" * 20)
                        # loss = ctc_loss报错 making sure all `forward` function outputs participate in calculating loss.
                        # 这导致计算att_loss的forward没有参与最终loss计算，改为
                        loss = ctc_loss + 0.0 * att_loss
                        att_loss = torch.tensor(0.0)
                    # loss_tuple = (ctc_weight*ctc_loss,(1-ctc_weight)*att_loss)
                    # print(loss, ctc_loss, att_loss)
                #                     print(f"joint ctc_att_loss={loss.item()}")
                else:
                    raise ValueError(
                        f"{model.encoder.config.ctc_loss_reduction} is not valid. Choose one of ['mean', 'sum']")
            else:
                if model.module.config.ctc_loss_reduction == "mean":
                    loss = loss.mean()
                elif model.module.config.ctc_loss_reduction == "sum":
                    # print(f"valid_num_labels={(inputs['labels'] >= 0).sum()}")
                    loss = loss.sum() / inputs["labels"].shape[0]
                    # loss = loss.sum() / (inputs['labels'] >= 0).sum()
                else:
                    raise ValueError(f"{model.config.ctc_loss_reduction} is not valid. Choose one of ['mean', 'sum']")
        else:
            # print("####################model is not wrapped####################")
            # print(f"####################type(model):{type(model)}####################")
            if self.eval_only_encoder:
                if model.encoder.config.ctc_loss_reduction == "mean":
                    ctc_loss = ctc_loss.mean()
                elif model.encoder.config.ctc_loss_reduction == "sum":
                    # print(f"valid_num_labels={(inputs['labels'] >= 0).sum()}")
                    ctc_loss = ctc_loss.sum() / inputs["labels"].shape[0]
                    #                     print(f"mean ctc_loss={loss}")
                    loss_tuple = (ctc_loss, att_loss)
                    loss = ctc_weight * ctc_loss + (1 - ctc_weight) * att_loss
                #                     print(f"joint ctc_att_loss={loss.item()}")
                else:
                    raise ValueError(
                        f"{model.encoder.config.ctc_loss_reduction} is not valid. Choose one of ['mean', 'sum']")
            else:
                if model.config.ctc_loss_reduction == "mean":
                    loss = loss.mean()
                elif model.config.ctc_loss_reduction == "sum":
                    # print(f"valid_num_labels={(inputs['labels'] >= 0).sum()}")
                    loss = loss.sum() / inputs["labels"].shape[0]
                else:
                    raise ValueError(f"{model.config.ctc_loss_reduction} is not valid. Choose one of ['mean', 'sum']")

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
            if self.eval_only_encoder:
                ctc_loss = ctc_weight * ctc_loss / self.args.gradient_accumulation_steps
                att_loss = (1 - ctc_weight) * att_loss / self.args.gradient_accumulation_steps
                loss_tuple = (ctc_loss, att_loss)
        else:
            if self.eval_only_encoder:
                ctc_loss = ctc_weight * ctc_loss / self.args.gradient_accumulation_steps
                att_loss = (1 - ctc_weight) * att_loss / self.args.gradient_accumulation_steps
                loss_tuple = (ctc_loss, att_loss)
        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            self.deepspeed.backward(loss)
        else:
            loss.backward()
        if self.eval_only_encoder:
            return loss.detach(), loss_tuple

        return loss.detach()


def show_args(args):
    print('\n'.join(['%s:%s' % item for item in args.__dict__.items()]))


def main():

    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    # parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    # model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(is_wandb_available)
    if is_wandb_available():
        import wandb
        wandb.init(project=training_args.logging_dir.split("/")[-1])
    configure_logger(model_args, training_args)
    stars = "*" * 20
    logger.info(f"{stars}model_args:{stars}\n{model_args}")
    logger.info(f"{stars}data_args:{stars}\n{data_args}")
    logger.info(f"{stars}training_args:{stars}\n{training_args}")
    processor = Wav2Vec2Processor.from_pretrained(model_args.processor_path)
    # encoder_config = AutoConfig.from_pretrained(
        # "model_config/config.json")
    # encoder_config.vocab_size = processor.tokenizer.vocab_size
    # model = Wav2Vec2ForCTC(encoder_config)
    model = Wav2Vec2ForCTC.from_pretrained(model_args.model_name_or_path)
    # state_dict = torch.load(f"{model_args.model_name_or_path}/pytorch_model.bin",map_location="cpu")
    # model.load_state_dict(state_dict,strict=False)
    if model_args.reinit_lm_head:
        nn.init.normal_(model.lm_head.weight)
        logger.info(f"{stars}reinitial lm_head{stars}")
    print(f"vocab_size = {processor.tokenizer.vocab_size}")
    assert processor.tokenizer.vocab_size == model.config.vocab_size,f"vocab_size of model is {model.vocab_size}, but vocab_size of model is {processor.tokenizer.vocab_size}"


    wer_metric = datasets.load_metric("/data2_from_58175/huggingface/metrics/wer")

    # prepare_train_path = "/data2_from_58175/huggingface/datasets/ws_train_s_test_meeting_test_net/train_s"
    # prepare_dev_path = "/data2_from_58175/huggingface/datasets/ws_train_s_test_meeting_test_net/test_meeting"
    dataset_path = data_args.dataset_path
    prepare_train_path = os.path.join(dataset_path, 'train')
    prepare_dev_path = os.path.join(dataset_path, 'dev')
    # prepare_train_path = "/data2_from_58175/huggingface/datasets/minnanyu/train"
    # prepare_dev_path = "/data2_from_58175/huggingface/datasets/minnanyu/test"
    logger.info(time.strftime('%Y-%m-%d %H:%M:%S'))
    logger.info(f"{stars}loading train_dataset{stars}")
    train_dataset = load_from_disk(prepare_train_path)

    ignored_columns = list(set(train_dataset.column_names) - set(["file", "text","length"]))
    train_dataset = train_dataset.remove_columns(ignored_columns)
    print(train_dataset[0]["file"])

    logger.info(time.strftime('%Y-%m-%d %H:%M:%S'))
    val_dataset = load_from_disk(prepare_dev_path)
    val_dataset = val_dataset.remove_columns(ignored_columns).select(range(500))

    # train_dataset = train_dataset.filter(lambda batch: float(0.25)<=batch["length"]<=float(15.0))
    # train_dataset.save_to_disk("hf_datasets/train_filt")

    train_total_dur = sum(train_dataset["length"]) / 3600
    train_maxlength = sorted(train_dataset["length"], reverse=True)[0]

    val_total_dur = sum(val_dataset["length"]) / 3600
    val_maxlength = sorted(val_dataset["length"], reverse=True)[0]

    logger.info(f"total_train_dataset:\n{train_dataset}\n")
    logger.info(f"total duration of train_dataset:\n{train_total_dur} hours\n")
    logger.info(f"maxlength of train_dataset:\n{train_maxlength} s\n")

    logger.info(f"val_dataset:\n{val_dataset}\n")
    logger.info(f"total duration of val_dataset:\n{val_total_dur} hours\n")
    logger.info(f"maxlength of val_dataset:\n{val_maxlength} s\n")

    # 变速在data_collator中实现，默认使用torchaudio，但由于一些原因，需要把speech值作为输入，而input_values值可以忽略，input_column_change在_remove_unused_columns作用
    logger.info(f"{stars}do speech perpturbation{stars}") if data_args.speed_perturb else None

    data_collator = DataCollatorCTCWithPadding_Speed_Perturb(processor=processor,
                                                             padding=True) if data_args.speed_perturb else DataCollatorCTCWithPadding(
        processor=processor, padding=True)
    # 验证集不进行在线增强
    data_collator_eval = DataCollatorCTCWithPadding(processor=processor, padding=True)

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        # gpt_tokenizer ,skip_special_tokens=True
        pred_str = processor.batch_decode(pred_ids)

        # we do not want to group tokens when computing the metrics
        # RUNNING ->RUNING
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        if logger.isEnabledFor(logging.INFO):
            for reference, predicted in zip(label_str[50:60], pred_str[50:60]):
                logger.info(f'reference: "{reference}"')
                logger.info(f'predicted: "{predicted}"')

        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    if model_args.freeze_feature_extractor:
        logger.info(f"{stars}freeze parameters of freeze_feature_extractor{stars} ")
        model.freeze_feature_extractor() if not model_args.encoder_decoder_mode else model.encoder.freeze_feature_extractor()
    if model_args.freeze_ALN:
        logger.info(f"{stars}freeze parameters of freeze_feature_extractor and feed_forward{stars} ")
        model.freeze_ALN() if not model_args.encoder_decoder_mode else model.encoder.freeze_ALN()
    if model_args.freeze_all_except_lm:
        logger.info(f"{stars}freeze all parameters of the model except lm_head{stars} ")
        model.freeze_all_except_lm() if not model_args.encoder_decoder_mode else model.encoder.freeze_all_except_lm()


    trainer = CTCTrainer(
        model=model,
        data_collator=data_collator,
        data_collator_eval=data_collator_eval,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor.feature_extractor,
        processor=processor,
        pseudo_model=None,
        pseudo_onthefly=False,
        teacher_model=None,
        eval_only_encoder=model_args.encoder_decoder_mode,
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)


if __name__ == "__main__":
    main()