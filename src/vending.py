"""
iVending: Seamless Smart Vending System

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
import ctypes
import cv2
import io
import multiprocessing as mp
import numpy as np
import os  # import os to set environmental variable
import picamera
import pickle
import platform
import pygame
import RPi.GPIO as GPIO  # import RPi GPIO package
import subprocess
import sys
import time
from billing import *
from edgetpu.basic.basic_engine import BasicEngine
from edgetpu.detection.engine import DetectionEngine
from face_detector import *
from math import pi
from picamera.array import PiRGBArray
from PIL import Image
from PIL import ImageDraw
from pyzbar import pyzbar
from pygame.locals import * # for event MOUSE variables
from servo_control import *
from tracker import *
import RPi.GPIO as GPIO  # import RPi GPIO package
import pigpio
import time  # import time to sleep
import picamera


# image streaming
lastresults = None
processes = []
frame_buffer = None
result_buffer = None
identity_dict = {}
fps = ""
detectfps = ""
framecount = 0
detectframecount = 0
time1 = 0
time2 = 0

# COnstant thresholds
TERMINATE_COUNTDOWN_THRESHOLD = 270
CONFID_THRESHOLD = 0.1
FACE_RECOG_THRESHOLD = 0.85

#Finite state machine
IDLE, TRACK, RESET = 0,1,2
state = None
current_identity = None
tracking_complete = None
terminate_countdown = TERMINATE_COUNTDOWN_THRESHOLD
MAX_DISAPPEAR_FRAMES = 40

# object tracking
trackers = {}
trackable_items = {}

# merchandises stock counts and price info 
items_dic = {'apple':2,'banana':3,'scissors':1,'bottle':2} 
price_dic = {'apple':1.25,'banana':1.0,'scissors':2.5,'bottle':1.75}

# shppoing cart, and cart buffer
cart = {}
cartBuffer = None

# eeceipt email html template 
email_template = None

# machine learning engines/models
od_engine = None
facenet_engine = None
face_detector = None
svm_clf = None

# define constants for various colors
WHITE = 255,255,255
BLACK = 0,0,0
GREEN = 0,255,0
YELLOW = 255,255,0

state_step_counter = 0
# pygame screen object
screen = None

end_program = None

# GPIO pin fo rthe quit button ont eh piTFT board
QUIT_PIN = 17

# display on piTFT
os.putenv('SDL_VIDEODRIVER', 'fbcon') 
os.putenv('SDL_FBDEV', '/dev/fb1')
# Track mouse clicks on piTFT
os.putenv('SDL_MOUSEDRV', 'TSLIB')
os.putenv('SDL_MOUSEDEV', '/dev/input/touchscreen')

# Set for broadcom numbering not board numbers...
GPIO.setmode(GPIO.BCM)
# setup quit button as input
GPIO.setup(QUIT_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)


"""
initialze the pygame screen and return the screen object
"""
def init_gui():
    # set initial PyGame variables
    pygame.init()
    size = width, height = 320, 240
    screen = pygame.display.set_mode(size)
    pygame.mouse.set_visible(False) # make mouse not visible
    screen.fill(BLACK) # Erase the work space
    return screen


"""
update the piTFT screen with different animation depending on the state
"""
def update_gui(screen, is_idle, step, current_identity, state):
    # check for pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit()
    screen.fill(BLACK)  # Erase the work space 
    if is_idle:
        # draw two arcs on piTFT on opposite side of a circle
        pygame.draw.arc(screen, YELLOW, [100, 20, 120, 120] , step*pi/8, (step+1)*pi/8, 4)
        pygame.draw.arc(screen, YELLOW, [100, 20, 120, 120] , (step+8)*pi/8, (step+9)*pi/8, 4)
        # set text font
        my_font = pygame.font.Font(None, 20)    
        text_surface = my_font.render("scanning...", True, WHITE)  # get text surface
        rect = text_surface.get_rect(center=[160,80]) # get rect objest of the text surface
        screen.blit(text_surface, rect) # blit text to scrren
        my_font = pygame.font.Font(None, 40)  
        text_surface = my_font.render("WELCOME!", True, WHITE)  # get text surface
        rect = text_surface.get_rect(center=[160,180]) # get rect objest of the text surface
        screen.blit(text_surface, rect) # blit text to scrren
    else:
        # draw two arcs on piTFT on opposite side of a circle
        pygame.draw.arc(screen, GREEN, [100, 60, 120, 120] , (16-step)*pi/8, (16-step+1)*pi/8, 4)
        pygame.draw.arc(screen, GREEN, [100, 60, 120, 120] , (16-step+8)*pi/8, (16-step+9)*pi/8, 4)
    # display workspace on screen (actual screen)
    pygame.display.flip()


"""
Clear the mp.Queue buffer by flushing out all the elements
"""
def clear_buffer(buffer):
    try:
        while not buffer.empty():
            buffer.get(False) # get a result off the queue immediately
        print("[Clearing]\n\t",buffer)
    except:
        return

"""
identify barcode in image frame, and verify identity found
"""
def scan_barcode(frame, identity_dict, current_identity):
    # find the barcodes in the frame and decode each of the barcodes
    barcodes = pyzbar.decode(frame)
    # loop over the detected barcodes
    for barcode in barcodes:
        # the barcode data is a bytes object so if we want to draw it
        # on our output image we need to convert it to a string first
        barcodeData = barcode.data.decode("utf-8")
        barcodeType = barcode.type
        # draw the barcode data and barcode type on the image
        print("Barcode Detected: {} ({})".format(barcodeData, barcodeType))
        # if the barcode text is found in database then start tracking, open lock, Green Light
        if barcodeData in identity_dict:
            print("\n=================================")
            print("Identity found: ", barcodeData, " ",identity_dict[barcodeData])
            print("=================================\n")
            current_identity.value = identity_dict[barcodeData][0] # ID


"""
This thread is responsible for coordinating the state transition of the program and handle any prerequisite and aftercare for each state. 
"""
def system_state_thread(running, state, frame_buffer, result_buffer, identity_dict, current_identity, tracking_complete, cartBuffer):
    global IDLE, TRACK, RESET
    global state_step_counter, screen
    global email_template
    #
    email_template = read_receipt_template()
    screen = init_gui()
    state_step_counter = 0
    try:
        while running.value:
            # Finite State Machine
            if state.value == IDLE:
                # camera_Thread need to be running and store image usable for qr or face
                # identify_Thread need to be running
                if current_identity.value > 0:
                    print("IDLE END")
                    clear_buffer(frame_buffer)
                    state.value = TRACK
                    state_step_counter = 0 # reset counter
                    # turn the camera facing down for tracking
                    turn_camera_down(ti=41000)
            elif state.value == TRACK:
                # stay in TRACK mode until tracking_complete flag is raised
                if tracking_complete.value:
                    state.value = RESET
                    state_step_counter = 0 # reset counter
            else: # state == RESET
                # clear buffers
                clear_buffer(frame_buffer)
                clear_buffer(result_buffer)
                state.value = IDLE
                # reset tracking_complete flag
                tracking_complete.value = False
                # get the cart of the latest transaction
                if not cartBuffer.empty():
                    last_cart = cartBuffer.get(False)
                    print("last_cart: ",last_cart)
                    receiver = {}
                    for pp in identity_dict.values():
                        if pp[0] == current_identity.value:
                            receiver['name'], receiver['email'] = pp[1], pp[2]
                    # email the receipt using cart info and customer identity
                    email_receipt(receiver, last_cart, price_dic, email_template)
                else:
                    print("No cart found")
                # clear buffers
                clear_buffer(cartBuffer)
                current_identity.value = 0
                # turn the camera facing up for identification scanning
                turn_camera_up(ti=54000)
            # sleep the thread, the state machine does't need to be update that rapidly
            time.sleep(0.1)

            # refresh screen graphic
            update_gui(screen, state.value == IDLE, state_step_counter, current_identity, state)

            state_step_counter += 1 if 
            # reset counter every 16 cycle, refresh at phase 1
            if state_step_counter%17 == 0:
                state_step_counter = 1
    except Exception as e:
        running.value = False
        print("[ERROR] end here: ", e)
    
    pygame.quit()
    print("[Finish] system_state_thread")


"""
This process is responsible for executing all image capture and processing tasks. 
"""
def image_processing_thread(running, state, label, result_buffer, frame_buffer, cartBuffer, identity_dict, current_identity, tracking_complete):
    global IDLE, TRACK, RESET
    global fps, detectfps, framecount, detectframecount, time1, time2
    global lastresults, camera_width, camera_height
    global trackers, trackable_items, cart

    camera_width,camera_height = 300, 300

    # initialize the cart and object tackers
    for item in items_dic.keys():
        trackers[item] = ObjectTracker(MAX_DISAPPEAR_FRAMES, True)
        cart[item] = 0

    with picamera.PiCamera() as camera:
        camera.framerate = 50
        # capture frames from the camera
        while running.value:
            if state.value == IDLE:
                camera.resolution = (720, 720)
                # image capture buffer
                rawCapture = PiRGBArray(camera, size=camera.resolution)
                # allow the camera to warmup
                time.sleep(0.3) 
                for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
                    # exit the loop when the program is terminating
                    if not running.value:
                        break
                    # break the loop when the state changed
                    if state.value != IDLE:
                        break
                    # obtain the frame array
                    frame = frame.array
                    if frame_buffer.full():
                        frame_buffer.get()
                    frame_buffer.put(frame.copy())
                    res = None
                    # get face detectiong result if available and then postprocess
                    if not result_buffer.empty():
                        res = result_buffer.get(False) # get a result off the queue immediately
                        imdraw = post_processing_image(frame, True, res, label, 720, 720)
                        frame = imdraw
                    # show the frame
                    cv2.imshow("Identification", frame)
                    key = cv2.waitKey(1) & 0xFF

                    # read bardcode from the frame
                    scan_barcode(frame, identity_dict, current_identity)

                    # clear the stream in preparation for the next frame
                    rawCapture.truncate(0)

            elif state.value == TRACK:
                camera.resolution = (camera_width, camera_height)
                # image capture buffer
                rawCapture = PiRGBArray(camera, size=camera.resolution)
                # allow the camera to warmup
                time.sleep(0.3)
            
                for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
                    t1 = time.perf_counter()
                    # exit the loop when the program is terminating
                    if not running.value:
                        break
                    # if state changed, save cart on the cart buffer
                    if state.value != TRACK:
                        # put cart data in buffer
                        cartBuffer.put(cart.copy())
                        # clean all the data related to the latest event cycle
                        for item in items_dic.keys():
                            trackers[item].clear()
                            cart[item] = 0
                        trackable_items.clear()
                        break
                    
                    # obtain the frame array
                    frame = frame.array
                    # if the buffer is full, pop the oldest item
                    if frame_buffer.full():
                        frame_buffer.get()
                    frame_buffer.put(frame.copy())

                    # post-process the latest result if available
                    res = None
                    if not result_buffer.empty():
                        res = result_buffer.get(False) # get a result off the queue immediately
                        detectframecount += 1
                        imdraw = post_processing_image(frame, False, res, label, camera_width, camera_height, tracking_complete)
                        lastresults = res
                    else:
                        imdraw = post_processing_image(frame, False, lastresults, label, camera_width, camera_height)

                    # show the frame
                    cv2.imshow("Tracking", imdraw)
                    if cv2.waitKey(1)&0xFF == ord('q'):
                        break

                    # clear the stream in preparation for the next frame
                    rawCapture.truncate(0)

                    # FPS calculation
                    framecount += 1
                    if framecount >= 15:
                        fps       = "(Playback) {:.1f} FPS".format(15/time1)
                        detectfps = "(Detection) {:.1f} FPS".format(detectframecount/time2)
                        framecount = 0
                        detectframecount = 0
                        time1 = 0
                        time2 = 0
                    t2 = time.perf_counter()
                    elapsedTime = t2-t1
                    time1 += elapsedTime
                    time2 += elapsedTime
                    #print("fps: ",fps,"\t|\tdetectfps: ",detectfps)
    cv2.destroyAllWindows()
    print("[Finish] image_processing_thread")


"""
This process is responsible for execution inference on all machine learning model utilizing the Edge TPU 
accelerator using the edgetpu library. The inputs are grab from the image frame queue, and the results will 
be pushed to the result queue for intra process communications.
"""
def inference_thread(running, state, result_buffer, frame_buffer, args, identity_dict, current_identity):
    global IDLE, TRACK, RESET, FACE_RECOG_THRESHOLD, FACE_RECOG_THRESHOLD_A
    global od_engine, face_detector, facenet_engine, svm_clf
    # Initialize object detection engine.
    od_engine = DetectionEngine(args.od_model)
    print("device_path: ", od_engine.device_path())
    _, od_width, od_height, _ = od_engine.get_input_tensor_shape()
    print("od input dim: ", od_width, od_height)
    # initial face detector using the opencv haarcascade model
    face_detector = FaceDetector(args.hc_model)
    # Initialize facenet engine.
    facenet_engine = BasicEngine(args.fn_model)
    # load the sklearn support vector machine model from disk
    svm_clf = pickle.load(open(args.svm_model, 'rb'))

    while running.value:
        # check if the frame buffer has a frame, else busy waiting
        if frame_buffer.empty():
            continue
        frame = frame_buffer.get()
        tinf = time.perf_counter()

        if state.value == IDLE:
            fd_results = None
            # reorder image frame from BGR to RGB
            img = frame[:,:,::-1]
            # face detection
            faces_coord = face_detector.detect(img, True)
            # image preprocessing, downsampling
            print("faces_coord: ",faces_coord)
            if not isinstance(faces_coord, type(None)):
                # normalize face image
                face_image = np.array(normalize_faces(img ,faces_coord))
                # facenet to generate face embedding
                facenet_engine.RunInference(face_image.flatten())
                face_emb = facenet_engine.get_raw_output().reshape(1,-1)
                # use SVM to classfy identity with face embedding
                pred_prob = svm_clf.predict_proba(face_emb)
                best_class_index = np.argmax(pred_prob, axis=1)[0]
                best_class_prob = pred_prob[0, best_class_index]
                print("best_class_index: ",best_class_index)
                print("best_class_prob: ",best_class_prob)
                print("label", svm_clf.classes_[best_class_index])
                # Check threshold and verify identify is in the identifiy dictionary
                if best_class_prob > FACE_RECOG_THRESHOLD:
                    face_label = svm_clf.classes_[best_class_index]
                    if face_label in identity_dict:
                        print("\n=================================")
                        print("Identity found: ", face_label, " ",identity_dict[face_label],
                            " with Prob = ", best_class_prob)
                        print("=================================\n")
                        current_identity.value = identity_dict[face_label][0] # ID
                result_buffer.put(faces_coord)
        elif state.value == TRACK:
            od_results = None
            # convert numpy array representation to PIL image with rgb format
            img = Image.fromarray(frame[:,:,::-1], 'RGB')
            # Run inference.
            od_results = od_engine.DetectWithImage(img, threshold=0.30, keep_aspect_ratio=True, relative_coord=False, top_k=10)
            # push result to buffer queue
            result_buffer.put(od_results)
        print(time.perf_counter() - tinf, "sec")
    print("[Finish] inference_thread")


"""
This function perform post-processing of image given the frame and the result from inference
"""
def post_processing_image(frame, fd, object_infos, label, camera_width, camera_height, tracking_complete=None):
    global trackers, trackable_items, cart, terminate_countdown, TERMINATE_COUNTDOWN_THRESHOLD
    if isinstance(object_infos, type(None)):
        return frame
    img_cp = frame.copy()

    # if facial detect
    if fd:
        # add image overlay of the face bonding boxes
        for face in object_infos:
            box = np.array(face).flatten().astype(int)
            cv2.rectangle(img_cp, (box[0],box[1]),(box[2],box[3]), (255,0,0), 2)
            # draw label above the rectangle
            y = box[1] - 15 if box[1] - 15 > 15 else box[1] + 15
            cv2.putText(img_cp, "face", (box[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
        return img_cp

    # create a dictionary to store rects for each items
    object_rects = {}
    # if a person is detected in the frame
    motion_presence = False
    for item in items_dic.keys():
        object_rects[item] = []

    for obj in object_infos:
        # creating overlay of bounding boxes
        if not label[obj.label_id] in items_dic:
            continue
        box = []
        box = obj.bounding_box.flatten().astype(int)
        # check if confience of detected objects is above threshold
        # rasie motion_presence if any object of person is detected 
        if label[obj.label_id] in items_dic and obj.score > CONFID_THRESHOLD:
            object_rects[label[obj.label_id]].append(box)
            motion_presence = True
        if label[obj.label_id] == 'person': 
            motion_presence = True
    if not isinstance(tracking_complete, type(None)):
        # if no person detected, decrement countdown
        if not motion_presence:
            terminate_countdown -= 1
            # if no person detected for 100 frame, tracking phase is complete
            if terminate_countdown <= 0:
                tracking_complete.value = True
                terminate_countdown = TERMINATE_COUNTDOWN_THRESHOLD
        else: # reset countcount
            terminate_countdown = TERMINATE_COUNTDOWN_THRESHOLD

    # update all tracked objects of each item catagory
    for item in object_rects.keys():
        # update tracking
        centroids, obj_rects, disap_obj_ids = trackers[item].update(object_rects[item])
        # loop over the tracked objects
        for (object_id, centroid) in centroids.items():
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(object_id)
            cv2.putText(img_cp, text, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(img_cp, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

            box = obj_rects[object_id]
            # draw object rectangle
            cv2.rectangle(img_cp, (box[0],box[1]),(box[2],box[3]), (0,0,255), 2)
            # draw label above the rectangle
            label_text = item #+ " (" + str(obj.score * 100) + "%)" 
            y = box[1] - 15 if box[1] - 15 > 15 else box[1] + 15
            cv2.putText(img_cp, label_text, (box[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

            # Get trackable object if possible
            itemID = item + str(object_id)
            ti = trackable_items.get(itemID, None)
            # if there is no existing trackable object, create one
            if ti is None:
                ti = TrackableItem(itemID, centroid, obj_rects[object_id])
            # add current frame data to trackable object history
            else:
                ti.centroids.append(centroid)
                ti.append_size(obj_rects[object_id])
            # store the trackable object in our dictionary
            trackable_items[itemID] = ti

        for object_id in disap_obj_ids:
            itemID = item + str(object_id)
            ti = trackable_items.get(itemID, None)

            # check to see if the object has been counted or not
            if not ti.incart:
                # compute the difference between the y-coordinate of the 
                # current centroid and the verage of all previous centroids
                y = [c[1] for c in ti.centroids[:-1]]
                y_direction = ti.centroids[-1][1] - np.mean(y)
                # if the direction is negative (indicating the object
                # is moving up) AND the centroid is above the center
                # line, count the object
                if y_direction < 0 and ti.centroids[-1][1] < camera_width // 2:
                    cart[item] += 1
                    ti.incart = True
                # if the direction is positive (indicating the object
                # is moving down) AND the centroid is below the
                # center line, count the object
                elif y_direction > 0 and ti.centroids[-1][1] > camera_width // 2:
                    cart[item] -= 1
                    ti.incart = False
            # store the trackable object in our dictionary
            trackable_items[itemID] = ti
        print('cart', cart)
    return img_cp


def main():
    print("cv2 version: ", cv2.__version__)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--od_model', help='File path of object detection Tflite model.', 
        default="/home/pi/final_project/models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite")
    parser.add_argument(
        '--od_label', help='File path of object detection label file.', 
        default="/home/pi/final_project/models/coco_labels.txt")
    parser.add_argument(
        '--id_data', help='File path of identity data file.', 
        default="/home/pi/final_project/id_data/id_data.txt")
    parser.add_argument(
        '--hc_model', help='File path of haarcascade detector file.', 
        default="/home/pi/final_project/models/haarcascade_frontalface_default.xml")
    parser.add_argument(
        '--fn_model', help='File path of facenet Tflite model.', 
        default="/home/pi/final_project/models/facenet_edgetpu.tflite")
    parser.add_argument(
        '--svm_model', help='File path of SVM identity classifer pickle file', 
        default="/home/pi/final_project/models/svm_model")

    args = parser.parse_args()

    # read the object detection labels from the text files.
    od_label = {}
    with open(args.od_label, 'r') as f:
        pairs = (l.strip().split(maxsplit=1) for l in f.readlines())
        od_label = dict((int(k), v) for k, v in pairs)

    # read the identity data from the text files.
    identity_dict = {}
    with open(args.od_label, 'r') as f:
        infos = (l.strip().split(maxsplit=3) for l in f.readlines())
        identity_dict = dict((ide, [i,name,email]) for ide, i, name, email in infos)
    
    # Initialize all mp.Queue buffer used for intra-process communication
    frame_buffer = mp.Queue(10)
    result_buffer = mp.Queue()
    cartBuffer = mp.Queue(3)
    # Initialize all shared variables used for intra-process communication
    running = mp.Value('b', True)
    state = mp.Value('i', IDLE)
    current_identity = mp.Value('i', 0)
    tracking_complete = mp.Value('b', False)

    # Activation of system state control thread
    p = mp.Process(target=system_state_thread,
                   args=(running, state, frame_buffer, result_buffer, identity_dict, current_identity, tracking_complete, cartBuffer),
                   daemon=True)
    p.start()
    processes.append(p)

    # Activation of streaming thread
    p = mp.Process(target=image_processing_thread,
                   args=(running, state, od_label, result_buffer, frame_buffer, cartBuffer, identity_dict, current_identity, tracking_complete),
                   daemon=True)
    p.start()
    processes.append(p)

    # Activation of inference thread
    p = mp.Process(target=inference_thread,
                   args=(running, state, result_buffer, frame_buffer, args, identity_dict, current_identity),
                   daemon=True)
    p.start()
    processes.append(p)

    # block until the quit button is pressed
    GPIO.wait_for_edge(QUIT_PIN, GPIO.FALLING)

    # cleaning all queue buffer to avoid deadlock
    clear_buffer(frame_buffer)
    clear_buffer(result_buffer)
    clear_buffer(cartBuffer)

    # toggle the running flag, so all threads will end
    running.value = False
    # end all processes
    for i in range(len(processes)):
        processes[i].join(1)
        print("joined p",i)
    # clean up environment before exiting
    hw_pi.stop()
    pygame.display.quit()
    pygame.quit()
    cv2.destroyAllWindows()
    GPIO.cleanup()  # cleanup GPIO

if __name__ == '__main__':
    main()