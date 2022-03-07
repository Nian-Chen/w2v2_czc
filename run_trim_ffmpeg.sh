#!/bin/bash
file_name=
index=
. /tsdata/kaldi_utils/parse_options.sh || exit 1;
# 关键词参数[option]和普通的位置参数不一样，关键参数可借助utils/parse_options.sh识别
# 但必须在utils/parse_options.sh之前先初始化想传入的参数
if [ $# -ne 0 ]; then
  # $#算的是普通位置参数的个数，本例中为0，因为用关键词参数更为精准
  echo "Usage: $0 [options]"
  echo "e.g.: run_trim.sh --wav.scp --index 0"
  echo "Options:"
  echo "  --file_name <file_name>            # file_name."
  echo "  --index <index>            # index of sub_file to run"
  exit 1
fi
file_name=$file_name
echo "log_file index:$index"
index=$((${index}-1))
sub_file_name=$file_name$index
num_lines=$(wc -l $sub_file_name|awk '{print $1}')
echo "processing sub_file:$sub_file_name"
echo "num_lines:$num_lines"


# 不要用while read line,管道读取有bug，会吃掉命令开头的字母
# while read line;
# do
  # # line_to_do=${line}
  # # echo ${line_to_do}
  # # ${line_to_do}
  # $line || exit 1; 
  # echo $line
# done  <$sub_file_name
for n in $(seq $num_lines); do
  line=$(sed -n ${n}p $sub_file_name);
#  echo $line;
  # 加上|| exit 1;保证每条line命令都成功执行
  $line|| exit 1; 
done

# sh $sub_file_name  || exit 1;
# tail -10 $sub_file_name|while IFS= read -r line; do
  # $line
  # echo $line
# done #< $sub_file_name