import requests
import os
import json
from tqdm import tqdm

# 定义基础API URL和初始偏移量
BASE_URL = "https://datasets-server.huggingface.co/rows?dataset=c4&config=en&split=train"
OFFSET = 0
LENGTH = 100

# 定义保存数据的本地文件路径
SAVE_PATH = "/ossmnt/c4_20G/c4_data_20G.json"

# 定义目标文件大小（20GB）
TARGET_SIZE = 20 * (1024**3)  # 20 GB in bytes

def query(offset, length):
    url = f"{BASE_URL}&offset={offset}&length={length}"
    response = requests.get(url)

    # 检查HTTP响应状态码
    if response.status_code != 200:
        print(f"Failed to fetch data at offset {offset}: HTTP {response.status_code}")
        return None

    try:
        return response.json()
    except JSONDecodeError:
        print(f"Invalid JSON response at offset {offset}")
        return None


# 初始化输出数据结构
output_data = {
    "type": "text_only",
    "instances": []
}

# 使用tqdm创建进度条
with tqdm(total=TARGET_SIZE, desc="Downloading", unit="B", unit_scale=True) as pbar:
    while True:
        # 获取数据
        data = query(OFFSET, LENGTH)
        
        if data is None or 'rows' not in data:
            print(f"Skipping data at offset {OFFSET} due to error or missing 'rows'")
        else:

            # 从数据中提取所需的文本内容
            texts = [{"text": item['row']['text']} for item in data['rows']]
            
            # 追加到输出数据结构
            output_data["instances"].extend(texts)
            
            # 更新进度条
            pbar.update(len(json.dumps(texts).encode('utf-8')))
        
        # 更新偏移量
        OFFSET += LENGTH
        
        # 如果达到目标大小，则停止循环
        if pbar.n >= TARGET_SIZE:
            break

# 保存数据到本地文件
with open(SAVE_PATH, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)
