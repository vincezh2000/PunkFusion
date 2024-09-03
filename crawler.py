import os
import time
import random

import requests
from bs4 import BeautifulSoup

output_dir = "/data/PunkFusion/crypto_data"
total_punks = 9999
def download_punk(punk_id):
    url = f"https://cryptopunks.app/cryptopunks/details/{punk_id}"

    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Punk {punk_id} details: {e}")
        return

    try:
        soup = BeautifulSoup(response.content, 'html.parser')

        # 获取Punk类型
        punk_type_element = soup.find('div', class_='col-md-10 col-md-offset-1 col-xs-12').find('h4').find('a')
        if punk_type_element:
            punk_type = punk_type_element.get_text(strip=True)
        else:
            raise ValueError("Punk type not found in the page")

        # 获取Attributes
        attributes = []
        attribute_sections = soup.find_all('div', class_='col-md-4')
        for section in attribute_sections:
            attribute_link = section.find('a')
            if attribute_link:
                attribute_name = attribute_link.get_text(strip=True)
                attributes.append(attribute_name)

        if not attributes:
            caption = f"{punk_type}, No Attribute"
        else:
            caption = f"{punk_type}, {', '.join(attributes)}"

        # 下载图片
        image_url = soup.find("meta", property="og:image")['content']
        image_response = requests.get(image_url)
        image_response.raise_for_status()

        image_path = os.path.join(output_dir, f"{punk_id}.png")
        with open(image_path, 'wb') as f:
            f.write(image_response.content)

        # 保存文本文件
        text_path = os.path.join(output_dir, f"{punk_id}.txt")
        with open(text_path, 'w') as f:
            f.write(caption)

        print(f"Downloaded Punk {punk_id}: {caption}")

    except Exception as e:
        print(f"Error processing Punk {punk_id}: {e}")

    # 添加随机延迟，避免过于频繁的请求
    time.sleep(random.uniform(1, 3))


# 主函数：下载所有Punk
for punk_id in range(total_punks):
    try:
        download_punk(punk_id)
    except Exception as e:
        print(f"Unexpected error with Punk {punk_id}: {e}")

print("All Punks downloaded.")
