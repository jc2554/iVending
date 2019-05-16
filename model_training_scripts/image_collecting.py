"""
script to collect training data for the SVM classifying with using 
picamea on the Raspberry Pi to continuous capturing images.

MIT License

Copyright (c) 2019 JinJie Chen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""


# import the necessary packages
import argparse
import cv2
import numpy as np
import os  # import os to set environmental variable
import picamera
import sys
import time
from picamera.array import PiRGBArray
from PIL import Image
from PIL import ImageDraw


"""
collecting num_img number of image using picamera, save image user the 
user_id folderdirectory
"""
def collect_images(num_img, index, user_id):
    with picamera.PiCamera() as camera:
        camera.resolution = (720, 720)
        camera.framerate = 30
        # allow the camera to warmup
        time.sleep(1)
        # frame buffer
        rawCapture = PiRGBArray(camera, size=camera.resolution)
        time.sleep(0.5)
        i = index
        # capture frames from the camera
        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True): #, resize=(width, height)):
            # obtain the frame array
            frame = frame.array
            cv2.imshow("Identification", frame)
            # trying saving the captured image
            try:
                print(cv2.imwrite('raw_image_data/'+ str(user_id) +'/'+str(i)+'.jpg', frame))
            except Exception as e:
                print ('[Error] image write: ',e)
            i += 1
            if i-index >= num_img:
                break
            # clear the stream in preparation for the next frame
            rawCapture.truncate(0)


parser = argparse.ArgumentParser()
parser.add_argument(
    '--n_img', help='number of images to collect.', default=1)
parser.add_argument(
    '--idx', help='Start index of image file name.', default=0)
parser.add_argument(
    '--user', help='user id')
args = parser.parse_args()
print(args)

collect_images(int(args.n_img), int(args.idx), args.user)