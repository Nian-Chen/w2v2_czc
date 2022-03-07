#!/bin/bash
data_dir=
processor_path=
. /tsdata/kaldi_utils/parse_options.sh || exit 1;
data_dir=$data_dir
processor_path=$processor_path
echo "1、get dict"
sh get_dict.sh --data_dir $data_dir
echo "2、get vocab.json and add it into $processor_path"
python get_processor.py --dict_path $data_dir/dict/lang_char.txt --processor_path $processor_path

