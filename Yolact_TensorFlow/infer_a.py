'''
Program for Yolact
Project for CMPE 258 - Deep Learning
By Aditya Sahu 7819
'''

import numpy as np
import tensorflow as tf
import cv2
import argparse as ap
import time

#function to check for valid option
def check_valid_option(option):
    while True:
        if option == 'video' or option == 'webcam':
            return option
        else:
            print("You have entered an Invalid input option, Enter again: ")


#taking input argment as video or webcam
parser = ap.ArgumentParser(description='Please select option: video or webcam')
parser.add_argument('-i', '--input', metavar='', type=str, required=True, help='Input type for yolo: video or webcam')
arguments = parser.parse_args()
option = check_valid_option(arguments.input)

#Giving the model path
saved_model_path = tf.saved_model.load('C:/Users/adity/PycharmProjects/aditya_sahu_yolacttf/yolact_tf/saved_model/')

infer = saved_model_path.signatures["serving_default"]

if option=='video':
    video_path = input('Please enter the file location\n')
    capture = cv2.VideoCapture(video_path)
    wx = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    hy = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    out_detection = cv2.VideoWriter('Aditya_Sahu_7819_cmpe258_yolact_video_output.mp4', fourcc, fps,(550, 550))


elif option =='webcam':
    capture = cv2.VideoCapture(0)
    wx = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    hy = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    out_detection = cv2.VideoWriter('Aditya_Sahu_7819_cmpe258_yolact_webcam_video_output.mp4', fourcc, fps,(550, 550))



str_time = time.time()
while capture.isOpened() and int(time.time() - str_time) < 100:
    ret, frame = capture.read(0)
    if not ret:
        print('Ret is false..')
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (550, 550)).astype(np.float32)
    output = infer(tf.constant(frame[None, ...]))

    _hy = frame.shape[0]
    _wx = frame.shape[1]


    detection_num = output['num_detections'][0].numpy()
    detection_boxes = output['detection_boxes'][0][:detection_num]
    detection_boxes = detection_boxes.numpy() * np.array([_hy, _wx, _hy, _wx])
    detection_masks = output['detection_masks'][0][:detection_num].numpy()
    detection_classes = output['detection_classes'][0][:detection_num].numpy()
    detection_scores = output['detection_scores'][0][:detection_num].numpy()

    for i in range(detection_num):
        score = detection_scores[i]
        if score > 0.5:
            box = detection_boxes[i].astype(int)
            _class = detection_classes[i]
            b0 = box[0]
            b1 = box[1]
            b2 = box[2]
            b3 = box[3]
            cv2.rectangle(frame, (b1, b0), (b3, b2), (0, 255, 0), 2)
            cv2.putText(frame, str(_class) + '; ' + str(round(score, 2)), (b1, b0), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 0), lineType=cv2.LINE_AA)
            mask = detection_masks[i]
            mask = cv2.resize(mask, (_wx, _hy))
            mask = (mask > 0.5)
            roi_frame = frame[mask]
            blended = roi_frame.astype("uint8")
            frame[mask] = blended * [0, 0, 1]

            frame=cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = (frame).astype(np.uint8)


    # Rendering the output
    out_detection.write(np.uint8(frame))
    cv2.imshow("Aditya Sahu",np.hstack([frame]))

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print('break')
        break

capture.release()
out_detection.release()
cv2.destroyAllWindows()
for i in range(1, 5):
    cv2.waitKey(1)
