import cv2
import os
import shutil

# 加载预训练的人脸检测模型
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(image_path):
    # 读取图片
    image = cv2.imread(image_path)
    # 转换为灰度图，因为人脸检测需要在灰度图上进行
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 检测人脸
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # 如果检测到人脸，返回True
    if len(faces) > 0:
        return True
    else:
        return False

def clean_images(source_folder, destination_folder):
    # 确保目标文件夹存在
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_folder):
        image_path = os.path.join(source_folder, filename)
        
        # 检查文件是否为图片
        if os.path.isfile(image_path) and image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            # 检测图片中是否有人脸
            if detect_faces(image_path):
                # 如果有人脸，复制到目标文件夹
                shutil.copy2(image_path, destination_folder)
                print(f"Copied {filename} to {destination_folder}")
            else:
                # 如果没有人脸，删除图片
                os.remove(image_path)
                print(f"Deleted {filename} as it does not contain a face.")

# 设置源文件夹和目标文件夹路径
source_folder = 'D:\python project\jaoben\image'  # 替换为你的图片文件夹路径
destination_folder = 'clean-human-image'  # 目标文件夹名称

# 开始清洗图片
clean_images(source_folder, destination_folder)