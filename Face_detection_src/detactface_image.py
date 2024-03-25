import cv2
import mediapipe as mp
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import glob
import argparse

def get_detector():
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)
    return face_detection

def detect_face(img):
    face_detection = get_detector()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)
    faces = []
    if results.detections:
        for detection in results.detections:
            box = detection.location_data.relative_bounding_box
            x, y, w, h = box.xmin, box.ymin, box.width, box.height
            faces.append((int(x*img.shape[1]), int(y*img.shape[0]), int(w*img.shape[1]), int(h*img.shape[0])))
    return faces

def get_face_mesh():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
    return face_mesh

def align_face(img, face):
    x, y, w, h = face
    h_img, w_img = img.shape[:2]
    img_rgb = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
    face_mesh = get_face_mesh()
    results = face_mesh.process(img_rgb)
    facial_pose = []
    if results.multi_face_landmarks:
        for landmark in results.multi_face_landmarks[0].landmark:
            facial_pose.append(((x + w * landmark.x)/w_img, (y + h * landmark.y)/h_img))
    return facial_pose

def align_face_original(img, face):
    x, y, w, h = face
    h_img, w_img = img.shape[:2]
    img_rgb = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
    face_mesh = get_face_mesh()
    results = face_mesh.process(img_rgb)
    facial_pose = []
    if results.multi_face_landmarks:
        for landmark in results.multi_face_landmarks[0].landmark:
            facial_pose.append(((x + w * landmark.x), (y + h * landmark.y)))
    return facial_pose

def count_pose(facial_pose):
    xs = [x for x, y in facial_pose]
    ys = [y for x, y in facial_pose]
    avg_x = sum(xs) / len(xs)
    avg_y = sum(ys) / len(ys)
    sum_dxdy = sum(abs(x - avg_x) + abs(y - avg_y) for x, y in facial_pose)
    return avg_x, avg_y, sum_dxdy

def safe_crop(img, x0, x1, y0, y1, if_padding=True):
    h, w = img.shape[:2]
    pad_left = -min(0, x0)
    pad_right = max(x1 - w, 0)
    pad_top = -min(0, y0)
    pad_bottom = max(y1 - h, 0)
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

avg_y_on_1024 = 0.6076579708466568 * 1024
standard_dxdy = 105.64138908808019 * 1024

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

def crop(img, face_centers):
    h, w = img.shape[:2]
    crops = []
    for avg_x, avg_y, scale in face_centers:
        x0 = round(avg_x - 511.5 * scale)
        x1 = round(avg_x + 511.5 * scale)
        y0 = round(avg_y - avg_y_on_1024 * scale)
        y1 = round(avg_y + (1024 - avg_y_on_1024) * scale)
        print(f"Crop coordinates: x0={x0}, x1={x1}, y0={y0}, y1={y1}")
        crops.append(safe_crop(img, x0, x1, y0, y1, True))
    return crops

def process_video(video_path, output_img):
    centers = []
    scales = []
    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameRate = total_frames // 50
    success, img = vidcap.read()
    frame_num = 0
    extracted_frames = 0
    base_filename = os.path.splitext(os.path.basename(video_path))[0]
    output_img_path = os.path.join(output_img, base_filename)
    if not os.path.exists(output_img_path):
        os.makedirs(output_img_path)

    while success and extracted_frames < 50:
        if frame_num % frameRate == 0:
            face_centers = get_center_and_scale(img)
            cropped_faces = crop(img, face_centers)
            for i, face in enumerate(cropped_faces):
                cv2.imwrite(f'{output_img_path}/frame_{extracted_frames}_face_{i}.jpg', face)
            extracted_frames += 1
        success, img = vidcap.read()
        frame_num += 1

def main(input_folder):
    output_img = "content/output_img"
    if not os.path.exists(output_img):
        os.makedirs(output_img)

    video_files = glob.glob(os.path.join(input_folder, '*.mp4'))
    for video_path in video_files:
        print(f"Processing video: {video_path}")
        process_video(video_path, output_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process videos to extract facial imagery.')
    parser.add_argument('input_folder', type=str, help='The folder path containing video files to process.')
    args = parser.parse_args()
    main(args.input_folder)