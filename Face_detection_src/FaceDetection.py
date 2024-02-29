import argparse
import cv2
import json
import mediapipe as mp
import numpy as np
import os

# 移动平均滤波器
class MovingAverageFilter:
    def __init__(self, window_size=3):
        self.window_size = window_size
        self.data = []

    def update(self, value):
        if value is None: 
            return None
        self.data.append(value)
        while len(self.data) > self.window_size:
            self.data.pop(0)
        if len(self.data) > 0:
            return float(np.mean(self.data))  
        else:
            return None

# Kalman 滤波器
class KalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    def update(self, coordX):
        measured = np.array([[np.float32(coordX)], [0]], np.float32)
        self.kf.correct(measured)
        predicted = self.kf.predict()
        return float(predicted[0][0])  

# 解析器参数
parser = argparse.ArgumentParser()
parser.add_argument('--input_video_name', type=str)
parser.add_argument('--start_time', type=str)
parser.add_argument('--end_time', type=str)
parser.add_argument('--maxcenter_speed', type=float, default=1)
parser.add_argument('--padding_ratio', type=float, default=0)
parser.add_argument('--filter_type', type=str, choices=["kalman", "ma"], default="ma")
args = parser.parse_args()

def get_frame_no(time_str, fps):
    h, m, s = map(int, time_str.split(":"))
    return int((h * 3600 + m * 60 + s) * fps)

cap = cv2.VideoCapture(args.input_video_name)
fps = cap.get(cv2.CAP_PROP_FPS)

start_frame_no = get_frame_no(args.start_time, fps)
end_frame_no = get_frame_no(args.end_time, fps)

maxcenter_speed = args.maxcenter_speed
padding_ratio = args.padding_ratio

if not os.path.exists("output"):
    os.makedirs("output")

box_position_file = open('output/box_position.jsonl', 'w')

mp_face_detection = mp.solutions.face_detection

# 根据参数选择滤波器
if args.filter_type == "kalman":
    center_x_filter = KalmanFilter()
    center_y_filter = KalmanFilter()
    size_filter = MovingAverageFilter()
elif args.filter_type == "ma":
    center_x_filter = MovingAverageFilter()
    center_y_filter = MovingAverageFilter()
    size_filter = MovingAverageFilter()

missing_face_counter = 0
current_center = None  
current_half_win_size = None  
pre_center = None

current_frame_no = 0
with mp_face_detection.FaceDetection(min_detection_confidence=0.7) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        current_frame_no += 1
        if current_frame_no < start_frame_no:
            continue

        elif current_frame_no > end_frame_no:
            break

        if "raw_clip_out" not in locals():
            raw_clip_out = cv2.VideoWriter('output/raw_clip.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                                           (image.shape[1], image.shape[0]))

        box_position_file.write(
            json.dumps({'current_center': str(current_center), 'current_half_win_size': str(current_half_win_size)}))

        results = face_detection.process(image)

        if results.detections:
            missing_face_counter = 0
            max_face_detection = max(results.detections,
                                     key=lambda detection: detection.location_data.relative_bounding_box.width *
                                                           detection.location_data.relative_bounding_box.height)

            center_x = max_face_detection.location_data.relative_bounding_box.xmin + \
                       max_face_detection.location_data.relative_bounding_box.width / 2
            center_y = max_face_detection.location_data.relative_bounding_box.ymin + \
                       max_face_detection.location_data.relative_bounding_box.height / 2

            if pre_center is None:
                pre_center = current_center = (center_x_filter.update(center_x), center_y_filter.update(center_y))
            else:
                _center = (center_x_filter.update(center_x), center_y_filter.update(center_y))
                diff = [_center[0] - pre_center[0], _center[1] - pre_center[1]]
                speed = abs(diff[0]*fps) if abs(diff[0]) >= abs(diff[1]) else abs(diff[1]*fps)
                if speed > maxcenter_speed:
                    print(f'Skipped a frame where speed ({speed}) was greater than the maximum speed ({maxcenter_speed}).')
                else:
                    current_center = _center
                pre_center = current_center

            half_win_size = max(max_face_detection.location_data.relative_bounding_box.width,
                                max_face_detection.location_data.relative_bounding_box.height) / 2 * (
                                        1 + padding_ratio)
            current_half_win_size = size_filter.update(half_win_size)

        else:
            missing_face_counter += 1
            if missing_face_counter > 5:
                missing_face_counter = 0
                current_center = None
                current_half_win_size = None

        if current_center is not None:
            start_x = int(current_center[0] * image.shape[1] - 256)
            start_y = int(current_center[1] * image.shape[0] - 256)
            end_x = int(current_center[0] * image.shape[1] + 256)
            end_y = int(current_center[1] * image.shape[0] + 256)
            face = image[max(start_y, 0):min(end_y, image.shape[0]), max(start_x, 0):min(end_x, image.shape[1])]
    
            resized_face = cv2.resize(face, (512, 512))

            if "cropped_face_out" not in locals():
                cropped_face_out = cv2.VideoWriter('output/cropped_face.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (512, 512))

            output_image = np.zeros((512, 512, 3), dtype=np.uint8)
            output_image[:(resized_face.shape[0]), :(resized_face.shape[1])] = resized_face
            cropped_face_out.write(output_image)

        raw_clip_out.write(image)

# 释放资源
cap.release()
raw_clip_out.release()
if "cropped_face_out" in locals():
    cropped_face_out.release()
box_position_file.close()
