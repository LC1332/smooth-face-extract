import os
import cv2

# 获取存储的所有人脸图片
face_images = os.listdir("face_images")

# 按文件名排序
face_images.sort(key=lambda x: int(x.split(".")[0].split("_")[1])) 

# 判断类别,获得编码器  
is_color = cv2.imread("face_images/" + face_images[0]).ndim == 3
if is_color:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG") 
else:
    fourcc = cv2.VideoWriter_fourcc(*"M", "J", "P", "G")

# 输出视频设置
fps = 30.0 
width = 640
height = 480
out_file = "combined_video.avi" 

# 创建VideoWriter对象
video_writer = cv2.VideoWriter(out_file, fourcc, fps, (width, height))

# 逐帧写入   
for image_name in face_images:
    frame = cv2.imread("face_images/" + image_name)
    video_writer.write(frame)
    
video_writer.release()