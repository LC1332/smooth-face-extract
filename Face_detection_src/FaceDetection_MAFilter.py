import cv2
import argparse
import mediapipe as mp
import json
import os
import numpy as np

class MovingAverageFilter:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.data = []

    def update(self, value):
        if value is None: 
            return None 
        self.data.append(value)
        while len(self.data) > self.window_size:
            self.data.pop(0)
        if len(self.data) > 0:
            return np.mean(self.data)
        else:
            return None

parser = argparse.ArgumentParser()
parser.add_argument('--input_video_name', type=str)
parser.add_argument('--start_time', type=str)
parser.add_argument('--end_time', type=str)
parser.add_argument('--maxcenter_speed', type=float, default=1)
parser.add_argument('--padding_ratio', type=float, default=0)
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

cap = cv2.VideoCapture(args.input_video_name)
fps = cap.get(cv2.CAP_PROP_FPS)

if not os.path.exists("output"):
    os.makedirs("output")

box_position_file = open('output/box_position.jsonl', 'w')

mp_face_detection = mp.solutions.face_detection

center_x_filter = MovingAverageFilter(window_size=3)
center_y_filter = MovingAverageFilter(window_size=3)
size_filter = MovingAverageFilter(window_size=3)

missing_face_counter = 0
current_center = None  # Add this line
current_half_win_size = None  # And this line

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success or cap.get(cv2.CAP_PROP_POS_FRAMES) > end_frame_no:
            break

        if "raw_clip_out" not in locals():  # Use of locals() to check the existence of variable
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

            current_center = (center_x_filter.update(center_x), center_y_filter.update(center_y))

            half_win_size = max(max_face_detection.location_data.relative_bounding_box.width,
                                max_face_detection.location_data.relative_bounding_box.height) / 2 * (1 + padding_ratio)
            current_half_win_size = size_filter.update(half_win_size)

        else:
            missing_face_counter += 1
            if missing_face_counter > 5:
                missing_face_counter = 0
                current_center = None
                current_half_win_size = None

        if current_center is not None and current_half_win_size is not None:
            start_x = int((current_center[0] - current_half_win_size) * image.shape[1])
            start_y = int((current_center[1] - current_half_win_size) * image.shape[0])
            end_x = int((current_center[0] + current_half_win_size) * image.shape[1])
            end_y = int((current_center[1] + current_half_win_size) * image.shape[0])
            face = image[max(start_y, 0):min(end_y, image.shape[0]), max(start_x, 0):min(end_x, image.shape[1])]

            face = cv2.resize(face, (640, 640))

            if "cropped_face_out" not in locals():  # Use of locals() to check the existence of variable
                cropped_face_out = cv2.VideoWriter('output/cropped_face.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                                   fps, (face.shape[1], face.shape[0]))
            cropped_face_out.write(face)

        raw_clip_out.write(image)  # Fix writing frame

    # Release everything if job is finished
    cap.release()
    raw_clip_out.release()
    if "cropped_face_out" in locals(): 
        cropped_face_out.release()
    box_position_file.close()