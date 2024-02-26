import cv2
import argparse
import mediapipe as mp
import json
import os
import numpy as np


class KalmanFilter:
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    def Estimate(self, coordX, coordY):
        ''' This function estimates the position of the object'''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        return predicted

# 参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--input_video_name', type=str)
parser.add_argument('--start_time', type=str)
parser.add_argument('--end_time', type=str)
parser.add_argument('--maxcenter_speed', type=float, default=1)
parser.add_argument('--padding_ratio', type=float, default=0)
args = parser.parse_args()

def get_frame_no(time_str):
    h, m, s = map(int, time_str.split(":"))
    return h * 3600 + m * 60 + s

# 通过时间获取开始和结束的帧
start_frame_no = get_frame_no(args.start_time)
end_frame_no = get_frame_no(args.end_time)

# 得到参数
maxcenter_speed = args.maxcenter_speed
padding_ratio = args.padding_ratio

#打开视频文件
cap = cv2.VideoCapture(args.input_video_name)
fps = cap.get(cv2.CAP_PROP_FPS)

# 设置当前的中心和窗口大小
current_center = None
current_half_win_size = None

# 跳转到开始的帧
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_no)

# 创建输出文件夹，如果不存在的话
if not os.path.exists("output"):
    os.makedirs("output")

# 创建保存视频的对象
raw_clip_out = cropped_face_out = None

# 创建用来保存滤波后的人脸检测框位置的文件
box_position_file = open('output/box_position.jsonl', 'w')

mp_face_detection = mp.solutions.face_detection
kfObj = KalmanFilter()

# 缺失人脸的帧计数器和速度限制
missing_face_counter = 0
speed_limit = args.maxcenter_speed

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
  while cap.isOpened():
    success, image = cap.read()
    if not success or cap.get(cv2.CAP_PROP_POS_FRAMES) > end_frame_no:
        break
    
    # Initialize the output video writers on the first frame read.
    if raw_clip_out is None:
        raw_clip_out = cv2.VideoWriter('output/raw_clip.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (image.shape[1], image.shape[0]))
        #raw_clip_out = cv2.VideoWriter('output/raw_clip.avi', cv2.VideoWriter_fourcc('H', '2', '6', '4'), fps, (image.shape[1], image.shape[0]))
    raw_clip_out.write(image)
    if cropped_face_out is None:
        cropped_face_out = cv2.VideoWriter('output/cropped_face.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (image.shape[1], image.shape[0]))
        #cropped_face_out = cv2.VideoWriter('output/cropped_face.avi', cv2.VideoWriter_fourcc('H', '2', '6', '4'), fps, (image.shape[1], image.shape[0]))

    results = face_detection.process(image)
    # raw_clip_out.write(image)

    if results.detections:
        #print("Detected faces in the frame")
        missing_face_counter = 0

        # 获得最大的脸的检测结果
        max_face_detection = max(results.detections, key=lambda detection: detection.location_data.relative_bounding_box.width * detection.location_data.relative_bounding_box.height)

        # 获得脸的中心位置
        center_x = max_face_detection.location_data.relative_bounding_box.xmin + max_face_detection.location_data.relative_bounding_box.width / 2
        center_y = max_face_detection.location_data.relative_bounding_box.ymin + max_face_detection.location_data.relative_bounding_box.height / 2

        # 使用Kalman filter来滤波中心位置
        predictedCoords = kfObj.Estimate(center_x, center_y)
        new_center = (predictedCoords[0, 0], predictedCoords[1, 0])

        if current_center is not None:
            # 计算速度，如果速度大于设定的值则跳过此帧
            diff_center = np.array(current_center) - np.array(new_center)
            speed = np.sqrt(np.sum(np.square(diff_center))) * fps
            if speed > maxcenter_speed:
                print(f'Skipped a frame where speed ({speed}) was greater than the maximum speed ({maxcenter_speed}).')
                continue

        # 更新当前的中心
        current_center = new_center

        # 获得半个窗口的大小并更新
        half_win_size = max(max_face_detection.location_data.relative_bounding_box.width, max_face_detection.location_data.relative_bounding_box.height) / 2 * (1 + padding_ratio)
        if current_half_win_size is None:
            current_half_win_size = half_win_size
        else:
            # 如果速度小于设定的值，更新窗口大小
            diff_size = half_win_size - current_half_win_size
            speed = abs(diff_size) * fps
            if speed < maxcenter_speed:
                current_half_win_size = half_win_size

        # 获取人脸并写入cropped_face_out
        start_x = int((current_center[0] - current_half_win_size) * image.shape[1])
        start_y = int((current_center[1] - current_half_win_size) * image.shape[0])
        end_x = int((current_center[0] + current_half_win_size) * image.shape[1])
        end_y = int((current_center[1] + current_half_win_size) * image.shape[0])
        face = image[max(start_y, 0):min(end_y, image.shape[0]), max(start_x, 0):min(end_x, image.shape[1])]
        
        # Resize face and write into cropped_face_out
        face = cv2.resize(face, (640,640))
        #print(f'Writing a face of size {face.shape} to cropped_face_out.')
        #print(f'Writing a face of size {face.shape} at frame {cap.get(cv2.CAP_PROP_POS_FRAMES)}')
        # cv2.imwrite(f'face_{cap.get(cv2.CAP_PROP_POS_FRAMES)}.png', face)
        cropped_face_out.write(face)

    else:
      missing_face_counter += 1
      if missing_face_counter > 5: # 你要跳过的帧数
        missing_face_counter = 0
        current_center = None
        current_half_win_size = None

    # 记录每帧的检测框的位置
    box_position_file.write(json.dumps({'current_center': str(current_center),
                                        'current_half_win_size': str(current_half_win_size)}))

# 释放资源
cap.release()
raw_clip_out.release()
cropped_face_out.release()
box_position_file.close()
