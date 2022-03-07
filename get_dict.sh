#!/bin/bash
data_dir=
. /tsdata/kaldi_utils/parse_options.sh || exit 1;
data_dir=$data_dir
dict=$data_dir/dict/lang_char.txt
echo "Make a dictionary"
mkdir -p $(dirname $dict)
echo "<pad> 0" > ${dict}
echo "<s> 1" >> ${dict}
echo "</s> 2" >> ${dict}
echo "<unk> 3" >> ${dict}
echo "| 4" >> ${dict}
python text2token.py -s 1 -n 1 --space "▁" $data_dir/text \
    | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -a -v -e '^\s*$' \
    | grep -v "▁" \
    | awk '{print $0 " " NR+2}' >> ${dict} \
    || exit 1;
num_token=$(cat $dict | wc -l)
echo "num_token = $num_token"
echo "output_file = $dict"

