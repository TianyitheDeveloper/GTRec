"""
python的命令行解析

2020年7月13日 16点53分
"""
# 导入命令行解析模块
import argparse
import sys

# 实例化一个对象
parse = argparse.ArgumentParser()

# 添加命令行
parse.add_argument("--boyfriend",type=str,default=0.01,help="who is Meisha's boyfriend?")
flags, unparsed = parse.parse_known_args(sys.argv[1:])

print("Meisha's boyfriend is ", flags.boyfriend)
print(flags)
print(unparsed)

