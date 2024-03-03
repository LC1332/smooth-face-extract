import requests
from bs4 import BeautifulSoup
import os
import time
import hashlib
import urllib.parse

# 定义起始网址
START_URL = 'https://highdefdiscnews.com/'

# 定义要抓取的特定文章页面的URL
ARTICLE_URL = 'https://highdefdiscnews.com/2022/03/25/ghostbusters-afterlife-4k-uhd-blu-ray-screenshots/'

# 定义图片URL的格式
IMAGE_URL_FORMAT = 'https://highdefdiscnews.com/wp-content/uploads/{}/{}'

# 定义图片下载目录
DOWNLOAD_DIR = 'downloaded_images2'

# 创建图片下载目录
if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)

# 下载图片的计数器
image_counter = 0

# 休眠功能的计时器
SLEEP_AFTER_IMAGES = 50
SLEEP_DURATION = 30

def get_md5_hash(url):
    """返回URL的MD5哈希值"""
    return hashlib.md5(url.encode()).hexdigest()

def download_image(image_url):
    """下载图片并保存为MD5哈希值命名的文件"""
    global image_counter
    try:
        response = requests.get(image_url, stream=True)
        if response.status_code == 200:
            # 获取图片内容的MD5哈希值作为文件名
            file_name = get_md5_hash(image_url) + '.png'
            file_path = os.path.join(DOWNLOAD_DIR, file_name)
            with open(file_path, 'wb') as file:
                file.write(response.content)
            print(f"图片下载完成: {file_path}")
            image_counter += 1
            # 检查是否需要休眠
            if image_counter % SLEEP_AFTER_IMAGES == 0:
                print(f"已下载 {SLEEP_AFTER_IMAGES} 张图片，休眠 {SLEEP_DURATION} 秒...")
                time.sleep(SLEEP_DURATION)
    except Exception as e:
        print(f"下载图片时出错: {e}")

def extract_images_from_article(article_url):
    """从文章页面提取并下载所有图片"""
    response = requests.get(article_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        for img_tag in soup.find_all('img'):
            img_url = img_tag.get('src')
            if img_url and img_url.startswith('https://highdefdiscnews.com/wp-content/uploads/'):
                download_image(img_url)

# 开始提取和下载图片
extract_images_from_article(ARTICLE_URL)