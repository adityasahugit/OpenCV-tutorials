# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 00:18:41 2019

@author: Aditya
"""

import cv2
import numpy as np

img = cv2.imread('photo_1.jpg')

grayscaled = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
th = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 1)
cv2.imshow('original',img)
cv2.imshow('Adaptive threshold',th)
cv2.waitKey(0)
cv2.destroyAllWindows()