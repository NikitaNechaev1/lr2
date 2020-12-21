# -*- coding: utf-8 -*-
"""
This scripts saves frames from video
"""

# Importing all necessary libraries
import cv2
import os
import sys

os.chdir(os.path.dirname(sys.argv[0])) # change working directory as script directory
current_dir = os.getcwd()
# Read the video from specified path
cam = cv2.VideoCapture(current_dir+'/video/six_string_samurai.avi')
folder = 'video_capture'

try:

    # creating a folder named data
    if not os.path.exists(folder):
        os.makedirs(folder)

    # if not created then raise error
except OSError:
    print('Error: Creating directory of data')

# frame
currentframe = 0
# how many frames to miss
miss_frame = 1
frame_num = 4

while True and currentframe < 50000:
    frame_num += 1    
    # reading from frame
    ret, frame = cam.read()

    if ret and frame_num%miss_frame == 0:
    # if ret:
        # if video is still left continue creating images
        name = './'+folder+'/frame' + str(currentframe) + '.jpeg'
        print('Creating...' + name)

        frame = cv2.resize(frame, (224, 224))

        # writing the extracted images
        cv2.imwrite(name, frame)

        # increasing counter so that it will
        # show how many frames are created
        currentframe += 1
    else:
        break

# Release all space and windows once done
cam.release()
cv2.destroyAllWindows()