import re

#读取txt文本文件
def readtxtfile(path="data/yuanzun.txt"):
    with open(path, 'r') as file:
    # 读取文件内容
        content = file.read()
    return content


def clean_json_string(s):
    str = s.find('[')
    end = s.refind(']')
    #print(f'clean {s[str:end + 1]}')
    return s[str:end + 1]