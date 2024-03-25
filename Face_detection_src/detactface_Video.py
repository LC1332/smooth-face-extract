#重构Python
import cv2
import mediapipe as mp
import os
import tqdm
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import glob
import shutil
import argparse

def get_video_files(directory, extensions=['.mp4', '.avi']):
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, '*' + ext)))
    return files

# 获取检测器Detect Face函数
def get_detector():
    mp_face_detection = mp.solutions.face_detection
    # 获取人脸检测模型
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)
    return face_detection

# 设置人脸检测模型
__face_detector = get_detector()

# 函数来检测人脸
def detect_face(img):
    face_detection = __face_detector
    # 将图像转换为RGB空间，模型期待它作为输入
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 获取模型结果
    results = face_detection.process(img_rgb)
    faces = []
    # 如果检测到面部，保存它们的边界框
    if results.detections:
        for detection in results.detections:
            box = detection.location_data.relative_bounding_box
            x, y, w, h = box.xmin, box.ymin, box.width, box.height
            faces.append((int(x*img.shape[1]), int(y*img.shape[0]), int(w*img.shape[1]), int(h*img.shape[0])))
    return faces

# 函数获取人脸网格（mesh）模型
def get_face_mesh():
    # 初始化
    mp_face_mesh = mp.solutions.face_mesh
    # 获取人脸网格模型
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
    return face_mesh

# 设置人脸网格模型
__face_mesh = get_face_mesh()

# 函数来对齐面部
def align_face(img, face):
    x, y, w, h = face
    h_img, w_img = img.shape[:2]
    img_rgb = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
    face_mesh = __face_mesh
    results = face_mesh.process(img_rgb)
    facial_pose = []
    # 如果检测到人脸标记，则保存他们
    if results.multi_face_landmarks:
        for landmark in results.multi_face_landmarks[0].landmark:
            facial_pose.append(((x + w * landmark.x)/w_img, (y + h * landmark.y)/h_img))
    return facial_pose

# 原始版本的align_face函数
def align_face_original(img, face):
    x, y, w, h = face
    # 确保坐标在图像内部
    h_img, w_img = img.shape[:2]
    x, y = max(0, x), max(0, y)
    w, h = min(w, w_img - x), min(h, h_img - y)

    if w <= 0 or h <= 0:
        print("Invalid face region.")
        return []

    face_img = img[y:y+h, x:x+w]
    # 检查裁剪后的图像是否为空
    if face_img.size == 0:
        print("Cropped face region is empty.")
        return []

    img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_mesh = __face_mesh
    results = face_mesh.process(img_rgb)

    facial_pose = []
    if results.multi_face_landmarks:
        for landmark in results.multi_face_landmarks[0].landmark:
            facial_pose.append(((x + w * landmark.x), (y + h * landmark.y)))
    return facial_pose

# 函数来计算姿势
def count_pose(facial_pose):

    xs = [x for x, y in facial_pose]
    ys = [y for x, y in facial_pose]
    avg_x = sum(xs) / len(xs)
    avg_y = sum(ys) / len(ys)
    sum_dxdy = sum(abs(x - avg_x) + abs(y - avg_y) for x, y in facial_pose)
    #print("sum_dxdy: ", sum_dxdy)
    #print(f'[Func 1] avg_x: {avg_x}, avg_y: {avg_y}, sum_dxdy: {sum_dxdy}')
    return avg_x, avg_y, sum_dxdy


# 函数剪切图像
def safe_crop(img, x0, x1, y0, y1, if_padding=True):
    #print(f'[Func 3] Crop area: x0={x0}, y0={y0}, x1={x1}, y1={y1}')
    h, w = img.shape[:2]
    pad_left = -min(0, x0)
    pad_right = max(x1 - w, 0)
    pad_top = -min(0, y0)
    pad_bottom = max(y1 - h, 0)
    #print(f'[Func 3] Padding: left={pad_left}, right={pad_right}, top={pad_top}, bottom={pad_bottom}')
    x0 = max(0, x0)
    x1 = min(w, x1)
    y0 = max(0, y0)
    y1 = min(h, y1)
    cropped_img = img[y0:y1, x0:x1]

    if if_padding and (pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0):
        if img.ndim == 3:
            pad_width = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
        else:
            pad_width = ((pad_top, pad_bottom), (pad_left, pad_right))
        mode = 'edge'
        cropped_img = np.pad(cropped_img, pad_width=pad_width, mode=mode)
    return cropped_img

# 函数获取面部中心和比例
def get_center_and_scale(img, minimal_scale_ratio=0.75):
    faces = detect_face(img)
    face_centers = []
    for face in faces:
        facial_pose = align_face_original(img, face)
        if facial_pose != []:
            avg_x, avg_y, sum_dxdy = count_pose(facial_pose)
            scale = sum_dxdy / standard_dxdy
            face_centers.append((avg_x, avg_y, scale))
    return face_centers

# 从图像中裁剪人脸的函数
def crop(img, face_centers):
    h, w = img.shape[:2]
    crops = []
    for avg_x, avg_y, scale in face_centers:
        x0 = round(avg_x - 511.5 * scale)
        x1 = round(avg_x + 511.5 * scale)
        y0 = round(avg_y - avg_y_on_1024 * scale )
        y1 = round(avg_y + (1024 - avg_y_on_1024)* scale )
        print(f"Crop coordinates: x0={x0}, x1={x1}, y0={y0}, y1={y1}")  # 打印裁剪坐标
        crops.append(safe_crop(img, x0, x1, y0, y1, True))
    return crops

# 简单低通滤波器
def lowpass_filter(y, alpha):
    '''简单的低通滤波器'''
    y_filtered = np.zeros_like(y)
    y_filtered[0] = y[0]
    for i in range(1, len(y)):
        y_filtered[i] = alpha * y[i] + (1 - alpha) * y_filtered[i-1]
    return y_filtered


def process_video(video_path, output_img_folder):
    # 解析原视频名称用于输出文件名
    base_video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_video_path = os.path.join('output_video', f'{base_video_name}_cropped_faces.mp4')

    vidcap = cv2.VideoCapture(video_path)
    success, img = vidcap.read()
    frame_num = 0
    centers = []
    scales = []
    cropped_faces = []
    w, h = None, None
    while success:
        faces = detect_face(img)
        face_centers = get_center_and_scale(img)
        cropped = crop(img, face_centers)
        for i, face in enumerate(cropped):
            cv2.imwrite(os.path.join(output_img_folder, f'frame_{frame_num}_face_{i}.jpg'), face)
        center = max(face_centers, key=lambda x: x[2]) if face_centers else None
        if center:
            centers.append(center[:2])
            scales.append(center[2])
        else:
            centers.append(centers[-1] if centers else (0, 0))
            scales.append(scales[-1] if scales else 1)
        success, img = vidcap.read()
        frame_num += 1
        cropped_faces.extend(cropped)

    if not cropped_faces:
        print("No faces detected in the video.")
        return

    h, w = cropped_faces[0].shape[:2]
    centers = np.array(centers)
    centers = medfilt(centers, kernel_size=3)
    scales = np.array(scales)
    scales = medfilt(scales, kernel_size=3)
    alpha = 0.1
    for i in range(2):
        centers[:,i] = lowpass_filter(centers[:,i], alpha)
    scales = lowpass_filter(scales, alpha)

    plt.figure(figsize=(10, 5))
    plt.plot(centers)
    plt.title('Face Centers vs. Time')
    plt.xlabel('Time (frames)')
    plt.ylabel('Face Center Coordinates')
    plt.legend(['x', 'y'])
    plt.savefig(os.path.join(output_img_folder, f'{base_video_name}_face_centers_vs_time.png'))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, 30, (w, h))
    for i in range(frame_num):
        for j in range(len(cropped_faces)):
            img_name = os.path.join(output_img_folder, f'frame_{i}_face_{j}.jpg')
            img = cv2.imread(img_name)
            if img is not None:
                video.write(cv2.resize(img, (w, h)))

    video.release()

    # 清理output_img_folder
    shutil.rmtree(output_img_folder)
    os.mkdir(output_img_folder)



# 修改main函数以处理视频
def main(folder_path):
    output_folder = "output_video"
    output_img = "output_img"
    if not os.path.exists(output_img):
        os.mkdir(output_img)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    global avg_y_on_1024
    global standard_dxdy
    avg_y_on_1024 = 0.6076579708466568 * 1024
    standard_dxdy = 105.64138908808019 * 1024

    video_files = get_video_files(folder_path)
    for video_file in video_files:
        try:
            process_video(video_file, output_img)
        except Exception as e:
            print(f"Error processing {video_file}: {e}")
            continue  # 中断当前视频处理，继续下一个视频



# 命令行接口
if __name__ == "__main__":
    # 创建Argument Parser对象
    parser = argparse.ArgumentParser(description="处理指定文件夹内的视频文件，提取面部信息。")

    # 添加文件夹路径参数
    parser.add_argument("folder_path", type=str, help="要处理的视频文件的文件夹路径。")

    # 解析命令行参数
    args = parser.parse_args()

    main(args.folder_path)