import cv2
import mediapipe as mp
import argparse
import json
import numpy as np
import os

#获取视频帧数
def get_frame_no(time_str, fps):
    h, m, s = map(int, time_str.split(":"))
    return int((h * 3600 + m * 60 + s) * fps)

#创建检测器实例
def get_face_detector():
    mp_face_detection = mp.solutions.face_detection
    return mp_face_detection.FaceDetection(min_detection_confidence=0.8)

#创建检测的窗口
def video_to_boxes( video_name, start_time, end_time ):
    ans = []
    cap = cv2.VideoCapture(video_name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame_no = get_frame_no(start_time, fps)
    end_frame_no = get_frame_no(end_time, fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_no)
    ret, frame = cap.read()

    face_detector = get_face_detector()

    while cap.isOpened():
        success, image = cap.read()
        frame_id = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if not success or frame_id > end_frame_no:
            break
        results = face_detector.process(image)
        ans.append(results)

    cap.release()

    return ans

#获取视频的长和宽
def get_video_shape( video_name ):
    cap = cv2.VideoCapture(video_name)

    ret, frame = cap.read()

    video_shape = frame.shape

    return video_shape[0], video_shape[1]
# video_height, video_width = get_video_shape(input_video_name)
# detection_results = video_to_boxes(input_video_name, start_time, end_time)



#找到全序列帧最大的人脸
def get_max_face_index(detection_results):
    maximal_area = -1
    max_frame_index = -1
    max_result = None
    for i, frame_result in enumerate(detection_results):
        if frame_result.detections:
            for result in frame_result.detections:
                area = result.location_data.relative_bounding_box.width * result.location_data.relative_bounding_box.height
                if area > maximal_area:
                    maximal_area = area
                    max_frame_index = i
                    max_result = result
    return max_frame_index, max_result
#max_frame_index, max_result = get_max_face_index(detection_results)


#不进行print了
# print(max_result)


#获取窗口大小
def get_winsize( data , video_height, video_width ):
    # return relative winsize to the height
    ans = data.location_data.relative_bounding_box.width
    ans += (data.location_data.relative_bounding_box.height * video_height / video_width)
    return ans / 2



#从前往后和从后往前去track人脸
def get_center_x( data ):
    return data.location_data.relative_bounding_box.xmin + data.location_data.relative_bounding_box.width/2

def get_center_y( data ):
    return data.location_data.relative_bounding_box.ymin + data.location_data.relative_bounding_box.height/2

#获取最大的人脸
def grab_one_face_from_detection( detection_results, max_frame_index, max_result ):
    n_frame = len(detection_results)
    maximal_location = max_result.location_data.relative_bounding_box

    last_result = max_result
    center_x = maximal_location.xmin + maximal_location.width / 2
    center_y = maximal_location.ymin + maximal_location.height / 2

    post_ans = []

    # 从前向后选取track的人脸
    for i in range( max_frame_index, n_frame ):
        if detection_results[i].detections:
            nearest_box = min( detection_results[i].detections, \
                              key=lambda x: abs( get_center_x(x) - get_center_x(last_result) ) \
                               + abs(get_center_y(x)  - get_center_y(last_result)) )
            post_ans.append( nearest_box )
            last_result = nearest_box
        else:
            post_ans.append( last_result )

    pre_ans = []

    last_result = max_result
    center_x = maximal_location.xmin + maximal_location.width / 2
    center_y = maximal_location.ymin + maximal_location.height / 2

    for i in range( max_frame_index-1, 0-1, -1 ):
        if detection_results[i].detections:
            nearest_box = min( detection_results[i].detections, \
                               key=lambda x: abs( get_center_x(x) - get_center_x(last_result) ) \
                               + abs(get_center_y(x)  - get_center_y(last_result)) )
            pre_ans.append( nearest_box )
            last_result = nearest_box
        else:
            pre_ans.append( last_result )

    ans = pre_ans[::-1] + post_ans

    return ans

#box per_frame怎么样去改

#box_per_frame = grab_one_face_from_detection(detection_results, max_frame_index, max_result)
 


#这里进行一般化处理以及设定MA的滑动窗口
#max_velocity = 14
def normalize_position( x, y , half_win_size , video_height, video_width ):
    half_win_size = min( 0.5, half_win_size )
    # y - half_win_size > 0
    y = max( y, half_win_size)
    # y + half_win_size < 1
    y = min( y, 1 - half_win_size )

    half_win_size_for_x = half_win_size * video_height / video_width
    # x - half_win_size_for_x > 0
    x = max( x, half_win_size_for_x )
    # x + half_win_size_for_x < 1
    x = min( x, 1 - half_win_size_for_x )

    return x, y, half_win_size

#MA滤波
def moving_average(data, window_size):
    return [np.mean(data[i-window_size+1:i+1]) if i>=window_size else np.mean(data[:i+1]) for i in range(len(data))]


#滤波
def filter_box(box_per_frame, video_height, video_width, max_velocity):

    max_velocity_in_x = max_velocity / video_width
    max_velocity_in_y = max_velocity / video_height

    datas = []
    for box in box_per_frame:
        datas.append({
            "x": get_center_x(box),
            "y": get_center_y(box),
            "half_win_size": get_winsize(box, video_height, video_width)/2.0
        })

    ans = [datas[0]]
    last_data = datas[0]

    for data in datas[1:]:
        x = data["x"]
        y = data["y"]
        half_win_size = data["half_win_size"]

        new_x = max( last_data["x"] - max_velocity_in_x, min( last_data["x"] + max_velocity_in_x, x ) )
        new_y = max( last_data["y"] - max_velocity_in_y, min( last_data["y"] + max_velocity_in_y, y ) )

        new_half_win_size = half_win_size + max( abs(new_x - x) , abs(new_y - y) )

        last_hws = last_data["half_win_size"]

        new_half_win_size = max( last_hws - 3*max_velocity_in_y, min( last_hws + 3*max_velocity_in_y, new_half_win_size ) )

        new_x, new_y, new_half_win_size = normalize_position( new_x, new_y, new_half_win_size, video_height, video_width )

        ans.append({
            "x": new_x,
            "y": new_y,
            "half_win_size": new_half_win_size
        })
        last_data = ans[-1]

    # 新加的MA
    x_values = [data['x'] for data in ans]
    y_values = [data['y'] for data in ans]

    window_size = 10
    smoothed_x_values = moving_average(x_values, window_size)
    smoothed_y_values = moving_average(y_values, window_size)

    for i in range(len(ans)):
        ans[i]['x'] = smoothed_x_values[i]
        ans[i]['y'] = smoothed_y_values[i]

    return ans

#创建这个窗口
#filtered_box = filter_box(box_per_frame, video_height, video_width)

#选取窗口裁切
def crop_video( video_name,  start_time, end_time, filtered_box ):
    ans = []
    cap = cv2.VideoCapture(video_name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame_no = get_frame_no(start_time, fps)
    end_frame_no = get_frame_no(end_time, fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_no)
    ret, frame = cap.read()

    face_detector = get_face_detector()

    index = 0
    cropped_face_out = None

    while cap.isOpened():
        success, image = cap.read()
        frame_id = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if not success or frame_id > end_frame_no:
            break
        if index > len(filtered_box) - 1:
            break

        if cropped_face_out is None:
          cropped_face_out = cv2.VideoWriter('output/cropped_face.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (640, 640))

        box = filtered_box[index]
        index += 1
        x = box["x"]
        y = box["y"]
        half_win_size_int = int( image.shape[0] * box["half_win_size"] )

        start_x = int( image.shape[1] * x ) - half_win_size_int
        start_y = int( image.shape[0] * y ) - half_win_size_int
        end_x = int( image.shape[1] * x ) + half_win_size_int
        end_y = int( image.shape[0] * y ) + half_win_size_int
        face = image[max(start_y, 0):min(end_y, image.shape[0]), max(start_x, 0):min(end_x, image.shape[1])]

        face = cv2.resize(face, (640, 640))
        cropped_face_out.write(face)


    cap.release()

    return ans


def main():
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()

    # 向解析器中添加参数
    parser.add_argument('--input_video_name', type=str)
    parser.add_argument('--start_time', type=str)
    parser.add_argument('--end_time', type=str, default="00:02:00")
    parser.add_argument('--max_velocity', type=float, default=14)
    #parser.add_argument('--window_size', type=int, default=10)
    
    # 解析参数
    args = parser.parse_args()

    if not os.path.exists('output'):
        os.mkdir('output')

    # 获取视频高度和宽度
    video_height, video_width = get_video_shape(args.input_video_name)

    # 获取人脸检测结果
    detection_results = video_to_boxes(args.input_video_name, args.start_time, args.end_time)

    # 获取检测到人脸最多的帧及其结果
    max_frame_index, max_result = get_max_face_index(detection_results)

    # 根据获取的最大人脸检测结果收集每个帧的box
    box_per_frame = grab_one_face_from_detection(detection_results, max_frame_index, max_result)

    # 对每个帧的box进行过滤
    filtered_box = filter_box(box_per_frame, video_height, video_width, args.max_velocity)

    # 根据过滤后的box剪裁视频帧
    crop_video(args.input_video_name, args.start_time, args.end_time, filtered_box)



if __name__ == '__main__':
    main()





#输出这个corp_video
#crop_video = crop_video(input_video_name, start_time, end_time, filtered_box)


# def main():
#     # 声明参数
#     parser = argparse.ArgumentParser()
#     # 解析器参数
#     parser.add_argument('--input_video_name', type=str)
#     parser.add_argument('--start_time', type=str)
#     parser.add_argument('--end_time', type=str)
#     parser.add_argument('--maxcenter_speed', type=float, default=1)
#     parser.add_argument('--padding_ratio', type=float, default=0)
#     parser.add_argument('--filter_type', type=str, choices=["kalman", "ma"], default="ma")
#     parser.add_argument('--max_velocity', type=float, default="14")
#     args = parser.parse_args()
#     args = parser.parse_args()
#     video_height, video_width = get_video_shape(input_video_name)
#     box_per_frame = grab_one_face_from_detection(args.detection_results, args.max_frame_index, args.max_result)
#     detection_results = video_to_boxes(input_video_name, start_time, end_time)
#     max_frame_index, max_result = get_max_face_index(detection_results)
#     box_per_frame = grab_one_face_from_detection(detection_results, max_frame_index, max_result)
#     crop_video = crop_video(input_video_name, start_time, end_time, filtered_box)
