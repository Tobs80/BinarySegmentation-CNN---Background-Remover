# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 18:12:37 2022

@author: Marto
"""

# DataFlair background removal
# import necessary packages
import os
import cv2
import numpy as np
import mediapipe as mp

#Load previously saved model
from keras.models import load_model
import matplotlib.pyplot as plt

SIZE = 256
def preprocess(img):
  #img = cv2.imread(img,0)
  
  img = cv2.resize(img,(SIZE,SIZE),interpolation = cv2.INTER_AREA)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  #print(img.shape)

  
  return(img)

def maskear(og_test_img):
    #test_img = images_folder[rand]
    #test_mask = masks_folder[rand]
    
    #og_test_img = read_image(test_img)
    og_test_img= preprocess(og_test_img)
    test_img = [og_test_img]
    test_img = np.array(test_img )
    test_img  = np.expand_dims(test_img , axis = 3)

    #print(test_img.shape)
    #Normalize images
    test_img  = test_img  /255.  #Can also normalize or scale using MinMax scaler

    
    prediction = (model.predict(test_img)[0,:,:,0] > 0.5).astype(np.uint8)
    #print(prediction.shape)
    return(prediction)


model = load_model("87bs.hdf5", compile=False)

# store background images in a list
image_path = '/'
images = os.listdir(image_path)
image_index= 0
bg_image = cv2.imread(image_path+images[image_index])
# initialize mediapipe which is used to see how a better quality binary mask looks like
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)# create videocapture object to access the webcam
cap = cv2.VideoCapture(0)
while cap.isOpened():
  _, frame = cap.read()
  # flip the frame to horizontal direction
  frame = cv2.flip(frame, 1)
  height , width, channel = frame.shape
  RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  # get the result
  results = selfie_segmentation.process(RGB)
  mask_cv = results.segmentation_mask
  # extract segmented mask
  mask = maskear(frame)
  mask = cv2.resize(mask,(640,480),interpolation = cv2.INTER_AREA)
  # show outputs
  cv2.imshow("mask cv ", mask_cv)
  cv2.imshow("mask", mask*255)
  cv2.imshow("Frame", frame)
  #frame_rs = cv2.resize(frame,(SIZE,SIZE),interpolation = cv2.INTER_AREA)
  masked = mask*frame[:,:,1]
  cv2.imshow("Mask on frame", masked)
  key = cv2.waitKey(1)
  if key == ord('q'):
        break
cv2. destroyAllWindows() 