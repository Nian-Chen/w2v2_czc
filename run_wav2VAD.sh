#!/bin/bash
time python wav2VAD.py  \
--model_path=/data2_from_58175/huggingface/models/wav2vec2_gpt2/encoder  \
--processor_path=/data2_from_58175/huggingface/models/wav2vec2-large-960h-lv60-self \
--audio_dictionary=/tsdata/diarization/voxconverse21_duke/DEV/audio \
--chunck_dur=4 \
--gpu_device=1 \
--save_file=/tsdata/wav2vad/test.npy
