from PIL import Image
import os

# 定义要删除的分辨率
invalid_resolutions = [
    (300, 190),
    (150, 190),
    (300, 300),
    (320, 453),
    (320, 402),
    (700, 700)
]

# 创建保存清洁图片的文件夹
output_folder = 'clean-human-image2'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 读取并筛选图片
input_folder = 'D:\python project\jaoben\clean-human-image'  # 替换为你的输入文件夹路径
for filename in os.listdir(input_folder):
    filepath = os.path.join(input_folder, filename)
    try:
        with Image.open(filepath) as img:
            # 获取图片分辨率
            width, height = img.size
            # 检查是否为有效分辨率
            if (width, height) not in invalid_resolutions:
                # 将符合条件的图片保存到新的文件夹中
                output_filepath = os.path.join(output_folder, filename)
                img.save(output_filepath)
                print(f"Saved {filename}")
    except Exception as e:
        print(f"Error processing {filename}: {e}")