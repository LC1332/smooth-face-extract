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

current_center = None
current_half_win_size = None

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_no)

if not os.path.exists("output"):
    os.makedirs("output")

raw_clip_out = cropped_face_out = None

box_position_file = open('output/box_position.jsonl', 'w')

mp_face_detection = mp.solutions.face_detection
kfObj = KalmanFilter()

missing_face_counter = 0
speed_limit = args.maxcenter_speed

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success or cap.get(cv2.CAP_PROP_POS_FRAMES) > end_frame_no:
            break

        if raw_clip_out is None:
            raw_clip_out = cv2.VideoWriter('output/raw_clip.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                                            (image.shape[1], image.shape[0]))
        raw_clip_out.write(image)
        
        if cropped_face_out is None:
          cropped_face_out = cv2.VideoWriter('output/cropped_face.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (640, 480))

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

            predictedCoords = kfObj.Estimate(center_x, center_y)
            new_center = (predictedCoords[0, 0], predictedCoords[1, 0])

            if current_center is not None:
                diff_center = np.array(current_center) - np.array(new_center)
                speed = np.sqrt(np.sum(np.square(diff_center))) * fps
                if speed > maxcenter_speed:
                    print(f'Skipped a frame where speed ({speed}) was greater than the maximum speed ({maxcenter_speed}).')
                    box_position_file.write(
                        json.dumps({'current_center': str(current_center), 'current_half_win_size': str(current_half_win_size)}))
                    box_position_file.write("\n")
                    continue

            current_center = new_center

            half_win_size = max(max_face_detection.location_data.relative_bounding_box.width,
                                max_face_detection.location_data.relative_bounding_box.height) / 2 * (1 + padding_ratio)
            if current_half_win_size is None:
                current_half_win_size = half_win_size
            else:
                diff_size = half_win_size - current_half_win_size
                speed = abs(diff_size) * fps
                if speed < maxcenter_speed:
                    current_half_win_size = half_win_size

        else:
            missing_face_counter += 1
            if missing_face_counter > 5:
                missing_face_counter = 0
                current_center = None
                current_half_win_size = None
                box_position_file.write("\n")

        if current_center is not None and current_half_win_size is not None:
            start_x = int((current_center[0] - current_half_win_size) * image.shape[1])
            start_y = int((current_center[1] - current_half_win_size) * image.shape[0])
            end_x = int((current_center[0] + current_half_win_size) * image.shape[1])
            end_y = int((current_center[1] + current_half_win_size) * image.shape[0])
            face = image[max(start_y, 0):min(end_y, image.shape[0]), max(start_x, 0):min(end_x, image.shape[1])]

            face = cv2.resize(face, (640, 640))
            cropped_face_out.write(face)

        box_position_file.write(
            json.dumps({'current_center': str(current_center), 'current_half_win_size': str(current_half_win_size)}))

    cap.release()
    raw_clip_out.release()
    cropped_face_out.release()
    box_position_file.close()