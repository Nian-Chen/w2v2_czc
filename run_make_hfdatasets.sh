#!/bin/bash
#time python make_hfdatasets.py  \
#--input_train_file=data_noseg/tmp/dataset  \
#--save_path=./hf_datasets \
#--is_ch
time python make_hfdatasets.py  \
--input_train_file=data_seg_train/tmp/dataset  \
--save_path=./hf_datasets \
--segments_mode \
--trim_audio_path=/data2_from_58175/Train_Ali_near \
--is_ch