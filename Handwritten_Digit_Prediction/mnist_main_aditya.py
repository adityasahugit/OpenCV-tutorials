'''
    This program uses trained Keras model to detect 4-digit ID on Live webcam or saved video
    Architecture: CNN Model
    Dataset: MNIST
    Author: Aditya Sahu
'''

import numpy as np
import cv2
import keras
import tensorflow
from keras.models import load_model


def extract_digit_fn(frame, rect, pad=10):
    x, y, w, h = rect
    cropped_digit = gray_img[y - pad:y + h + pad, x - pad:x + w + pad]

    cropped_digit = cropped_digit / 255.0  # normalization
    if cropped_digit.shape[0] >= 32 and cropped_digit.shape[1] >= 32:
        cropped_digit = cv2.resize(cropped_digit, (28, 28))
    else:
        return
    return cropped_digit;


def preprocessing_fn(frame, tresh=90):
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    gray_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,blockSize=321, C=28)
    return gray_img


def canny_fn(frame, tresh=90):
    grayimage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cannyimage = cv2.Canny(grayimage, 100, 200)
    return cannyimage


def grayscale_fn(frame, tresh=90):
    grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return grayscale_img


capture = 0
out = 0

model = load_model("Aditya_Sahu_7819_model.h5")   #loaded model wich was trained in ModelTrain_mnist.py

print("Please enter '0' for live webcam or '1' for using saved video file ")
user_input = int(input())


if user_input == 0:
    capture = cv2.VideoCapture(0)
    frame_w = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) #1280
    frame_h = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) #720
    fps = capture.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter('webcam_video_mnist_output.mp4', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_w, frame_h))
    out_canny = cv2.VideoWriter('webcam_video_mnist_canny_output.mp4', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_w, frame_h))
    out_contour = cv2.VideoWriter('webcam__video_mnist_contour_output.mp4', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_w, frame_h))
    out_grayscale = cv2.VideoWriter('webcam_video_mnist_grayscale_output.mp4', cv2.VideoWriter_fourcc(*'XVID'), fps,(frame_w, frame_h))

elif user_input == 1:
    capture = cv2.VideoCapture('aditya_id_input.mp4')
    frame_w = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    print(frame_w,frame_h,fps)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print('duration sec- ' ,frame_count/fps)
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out = cv2.VideoWriter('mnist_output.mp4', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_w, frame_h))
    out_canny = cv2.VideoWriter('mnist_canny_output.mp4', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_w, frame_h))
    out_contour = cv2.VideoWriter('mnist_contour_output.mp4', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_w, frame_h))
    out_grayscale = cv2.VideoWriter('mnist_grayscale_output.mp4', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_w, frame_h))

capture.set(3, 5*128)   # size for webcam
capture.set(4, 5*128)

SIZE = 28  #size of image

before_cropped_digit=0

for i in range(1000):
    ret, frame = capture.read(0)
    gray_img = preprocessing_fn(frame)
    orig_image = frame

    canny_image=canny_fn(frame)
    grayscale_img=grayscale_fn(frame)
    contours, _ = cv2.findContours(gray_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rectangles = [cv2.boundingRect(contour) for contour in contours]
    rectangles = [rect for rect in rectangles if rect[2] >= 3 and rect[3] >= 8]

    w2 = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    h2 = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pre_image = np.ones((h2, w2), np.uint8) + 255
    pre_image.fill(255)
    pre_image_to_bgr = cv2.cvtColor(pre_image,cv2.COLOR_GRAY2BGR)
    for rect in rectangles:
        x, y, w, h = rect

        if i >= 0:
            org_frame = extract_digit_fn(frame, rect, pad = 15)
            
            if org_frame is not None:
                org_frame = np.expand_dims(org_frame, 0)
                org_frame = np.expand_dims(org_frame, 3)
                
                result_arr = model.predict(org_frame)
                class_prediction = np.argmax(result_arr, axis=1)

                cv2.rectangle(orig_image, (x, y), (x + w, y + h), color = (0, 255, 0), thickness=4)
                label = str(class_prediction)
                cv2.putText(orig_image, label, (rect[0]+20, rect[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                max_wh = max(w,h)
                square_bg = np.zeros((max_wh, max_wh), np.uint8)
                square_bg2 = np.zeros((max_wh, max_wh, 3), np.uint8)

                w_start = int((max_wh-w)/2)
                h_start = int((max_wh-h)/2)
                w_end = int((max_wh+w)/2)
                h_end = int((max_wh+h)/2)
                cropped_gray_img = gray_img[y:y + h, x:x + w]
                cropped_img = frame[y:y + h, x:x + w]
                square_bg[h_start:h_end, w_start:w_end] = cropped_gray_img.copy()
                square_bg2[h_start:h_end, w_start:w_end] = cropped_img.copy()
                try:
                    pre_image[y: y+max_wh, x: x+max_wh] = square_bg.copy()
                    orig_image[y+200: y+200+max_wh, x: x+max_wh] = square_bg2.copy()
                    pre_image_to_bgr = cv2.cvtColor(pre_image,cv2.COLOR_GRAY2BGR)
                except:
                    pass

    out.write(orig_image)
    out_contour.write(gray_img)
    out_grayscale.write(grayscale_img)
    out_canny.write(canny_image)
    # orig_image = cv2.resize(orig_image, (1000, 1000))
    # imSS = cv2.resize(pre_image_to_bgr, (640, 640))
    cv2.imshow("Aditya_bitwise",  pre_image_to_bgr)
    cv2.imshow("Aditya_Prediction",  orig_image)
    cv2.imshow("Aditya_Contour",gray_img)
    cv2.imshow("Aditya_Canny",canny_image)
    cv2.imshow("Aditya_GrayScale",grayscale_img)
    if cv2.waitKey(1) & 0xFF==ord("q"):
        print('break')
        break

capture.release()
out.release()
out_contour.release()
out_grayscale.release()
out_canny.release()
cv2.destroyAllWindows()
print("Predicted digits: ", "7,8,1,9")
print('Destroyed windows')