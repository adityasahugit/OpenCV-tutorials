import cv2
import numpy as np
import argparse as argp
import time
import tensorflow as tf

def check_user_option(user_input):
    while True:
        if user_input == 'webcam' or user_input == 'video':
            return user_input
        else:
            print("Invalid input option entered: ")
            print("Enter again")

parser = argp.ArgumentParser(description='Please give a option webcam or video for Yolact processing')
parser.add_argument('-i', '--input', metavar='', type=str, required=True,
                    help='Input type : video or webcam')
args = parser.parse_args()
option = check_user_option(args.input)
model = tf.saved_model.load('C:/Users/adity/PycharmProjects/aditya_sahu_yolacttf/yolact_tf/saved_model/')


infer = model.signatures["serving_default"]

if option =='webcam':
    capture = cv2.VideoCapture(0)
    w = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    output_detection = cv2.VideoWriter('Aditya_Sahu_7819_cmpe258_yolact_webcam_output.mp4', fourcc, fps,
                                       (550, 550))

elif option=='video':
    video_path = input('Please enter the file location\n')
    capture = cv2.VideoCapture(video_path)
    w = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    # output_detection = cv2.VideoWriter('Aditya_Sahu_7819_cmpe258_yolact_video_output.mp4', fourcc, fps,(550, 550))



start_time = time.time()
while capture.isOpened() and int(time.time() - start_time) < 100:
    ret, frame = capture.read(0)
    if not ret:
        print('Ret is false..')
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (550, 550)).astype(np.float32)
    output = infer(tf.constant(frame[None, ...]))

    _h = frame.shape[0]
    _w = frame.shape[1]


    det_num = output['num_detections'][0].numpy()
    det_boxes = output['detection_boxes'][0][:det_num]
    det_boxes = det_boxes.numpy() * np.array([_h, _w, _h, _w])
    det_masks = output['detection_masks'][0][:det_num].numpy()

    det_scores = output['detection_scores'][0][:det_num].numpy()
    det_classes = output['detection_classes'][0][:det_num].numpy()

    for i in range(det_num):
        score = det_scores[i]
        if score > 0.5:
            box = det_boxes[i].astype(int)
            _class = det_classes[i]
            cv2.rectangle(frame, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)
            cv2.putText(frame, str(_class) + '; ' + str(round(score, 2)), (box[1], box[0]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 0), lineType=cv2.LINE_AA)
            mask = det_masks[i]
            mask = cv2.resize(mask, (_w, _h))
            mask = (mask > 0.5)
            roi = frame[mask]
            blended =roi.astype("uint8")
            frame[mask] = blended * [0, 0, 1]
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = (frame * 255).astype(np.uint8)

    # Render the output
    output_detection.write(np.uint8(frame))
    cv2.imshow("Aditya Sahu",np.hstack([frame]))

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print('break')
        break

capture.release()
output_detection.release()
cv2.destroyAllWindows()
for i in range(1, 5):
    cv2.waitKey(1)