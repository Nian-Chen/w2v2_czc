#!/bin/bash
dataset_path=
trim_audio_outputdir=
. /tsdata/kaldi_utils/parse_options.sh || exit 1;
dataset_path=$dataset_path
trim_audio_outputdir=$trim_audio_outputdir
mkdir -p $trim_audio_outputdir || exit 1;
num_lines=$(wc -l $dataset_path|awk '{print $1}')
echo "num_lines : $num_lines"
echo "dataset_path : $dataset_path"
echo "trim_audio_outputdir : $trim_audio_outputdir"
ouput_file=$(dirname $dataset_path)/run_trim
awk -F'\t' -vprefix=$trim_audio_outputdir/ '{print "ffmpeg -loglevel quiet -y -i",$2,"-ss",$3,"-t",$5,prefix$1".flac"}' $dataset_path >$ouput_file
result_num_lines=$(wc -l $ouput_file|awk '{print $1}')
if [ $result_num_lines -eq $num_lines ]
then
    echo "output file : $ouput_file"
else
    echo "failed! num_lines of $ouput_file is not equal to that of $dataset_path" && exit 1
fi

subfile_num_lines=$[result_num_lines/10]
if [ $[result_num_lines%10] > 0 ]
then
  subfile_num_lines=$[subfile_num_lines+1]
fi
echo "splitting $ouput_file into 10 sub_files"
echo "lines of each subfile: $subfile_num_lines"

split -l $subfile_num_lines $ouput_file $ouput_file -d -a 1