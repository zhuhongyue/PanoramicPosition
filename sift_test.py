import numpy as np
import cv2
import scipy.misc
import dataloader

(train_lbl, train_img) = dataloader.readData("./train_image/scene1_jiading_lib_training_coordinates.csv")

img = cv2.imread('./pic/6987.jpg')

gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)

img=cv2.drawKeypoints(gray,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
scipy.misc.imsave('sift_file.jpg', img)
