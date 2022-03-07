import argparse
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="get processor, i.e. add vocab.json to processor_path")
    parser.add_argument("-dp", "--dict_path", type=str, default=None, help="the dict_path acquired by get_dict.sh")
    parser.add_argument("-pp", "--processor_path", type=str, default="processor", help="the processor_path")
    args = parser.parse_args()
    print(f"args = {args}")
    dict_path = args.dict_path
    processor_path = args.processor_path
    dict = {}
    with open(dict_path, 'r') as f:
        for index,line in enumerate(f):
            # 在词表开头加了几个特殊符号，保证1是代表<s>，2代表</s>
            # 因此需要借助index，而不是直接char_dict[int(arr[1])] = arr[0]
            arr = line.strip().split()
            assert len(arr) == 2
            dict[arr[0]] = int(index)
    with open(f'{processor_path}/vocab.json', 'w', encoding='utf_8') as vocab_file:
        json.dump(dict, vocab_file, ensure_ascii=False)