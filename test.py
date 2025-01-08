# 文件路径
file_path = '/home/yjy/heteroPrompt/ggfm/requirement.txt'

# 读取文件并将每一行作为一个元素存入列表
with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# 移除每行的换行符（\n）
lines = [line.strip() for line in lines]

# 输出列表
print(lines)
