"""
script to post-process training images by using OpenCV face detection 
and normalization

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

import argparse
import cv2
import numpy as np
import os


"""
process all image in the user_id subdirectory , save processed images in the 
user_id folderdirectory
"""
def process_images(user_id):
    images = []
    labels = []
    labels_dic = {}
    # list of people (subdirectory folder names)
    people = [person for person in os.listdir("raw_image_data/")] if user_id == -1 else [str(user_id)]
    count = 0
    for i, person in enumerate(people):
        labels_dic[i] = person
        image_names = [image for image in os.listdir("raw_image_data/" + person)]
        if not os.path.exists('image_data/'+person):
            os.makedirs('image_data/'+person)
        for j, image_name in enumerate(image_names):
            image = cv2.imread("raw_image_data/" + person + '/' + image_name, 1)
            images.append(image)
            labels.append(person)
            # face deection using the openCV Cascade Classifier
            scale_factor = 1.2
            min_neighbors = 5
            min_size = (5, 5)
            biggest_only = True
            faces_coord = classifier.detectMultiScale(image,
                                                      scaleFactor=scale_factor,
                                                      minNeighbors=min_neighbors,
                                                      minSize=min_size,
                                                      flags=cv2.CASCADE_SCALE_IMAGE)

            if not isinstance(faces_coord, type(None)):
                faces = normalize_faces(image ,faces_coord)
                cv2.imwrite('image_data/'+person+'/%s.jpeg' % (j), faces[0])
                count += 1
    print("Number of face image Generated: ", count)
    return (images, np.array(labels), labels_dic)


"""
Normalize image by
Truncate out the face from teh image using the bounding box
Resize the image with interpolation using openCv
"""
def normalize_faces(image, faces_coord, size=(160, 160)):
    faces = []
    # cut image by the bounding box
    for (x, y, w, h) in faces_coord:
        w_rm = int(0.3 * w / 2)
        faces.append(image[y: y + h, x + w_rm: x + w - w_rm])

    images_norm = []
    #resize image
    for face in faces:
        if image.shape < size:
            image_norm = cv2.resize(face, size, interpolation=cv2.INTER_AREA)
        else:
            image_norm = cv2.resize(face, size, interpolation=cv2.INTER_CUBIC)
        images_norm.append(image_norm)
    return images_norm

  

parser = argparse.ArgumentParser()
parser.add_argument(
    '--user', help='user id, -1 for all')
args = parser.parse_args()
print(args)

classifier = cv2.CascadeClassifier("../src/models/haarcascade_frontalface_default.xml")

images, labels, labels_dic = process_images(args.user)

print("num images: ", len(images))
print("labels_dic: ", labels_dic)
