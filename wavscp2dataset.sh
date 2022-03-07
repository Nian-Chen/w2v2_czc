#!/bin/bash

data_dir=
. kaldi_utils/parse_options.sh || exit 1;
data_dir=$data_dir
# 根据kaldi的准备文件进行改造，将segments改造为包含'id' 'filepath' 'length' 'text'的dataset，用'\t'隔开
file_needed="text wav.scp"
for file in ${file_needed}; do
    if [[ ! -f $data_dir/$file ]]; then
        echo "$data_dir/$file is needed" && exit 1;
    fi
done
num_lines=$(wc -l $data_dir/text|awk '{print $1}')
echo "num_lines : $num_lines"
mkdir -p $data_dir/tmp
cat $data_dir/wav.scp|awk '{print $1}' > $data_dir/tmp/onlyid
cat $data_dir/wav.scp|awk '{print $2}' > $data_dir/tmp/onlyfilepath
# 没有utt2dur就借助soxi生成一个
if [[ ! -f $data_dir/utt2dur ]]; then
    echo "generating utt2dur"
    for n in $(seq $num_lines); do
      line=$(sed -n ${n}p $data_dir/wav.scp) || exit 1;
      # ${line#* }删除左边起第一个空格及其左边部分
      dur=$(soxi -D ${line##* })
      # {line%% *}删除右边起最后一个空格及其右边部分
      new_line=${line%% *}' '$dur
      echo $new_line
    done >$data_dir/utt2dur
fi
cat $data_dir/utt2dur|awk '{print $2}' > $data_dir/tmp/onlydur
cat $data_dir/text|awk '{$1="";sub(/^[[:space:]]/,"");print $0}' > $data_dir/tmp/onlytext
ouput_file=$data_dir/tmp/dataset
paste -d "\t" $data_dir/tmp/onlyid $data_dir/tmp/onlyfilepath $data_dir/tmp/onlydur $data_dir/tmp/onlytext > $ouput_file
result_num_lines=$(wc -l $ouput_file|awk '{print $1}')
if [ $result_num_lines -eq $num_lines ]
then
    echo "output file : $ouput_file"
else
    echo "failed! num_lines of $ouput_file is not equal to that of $data_dir/text" && exit 1
fi