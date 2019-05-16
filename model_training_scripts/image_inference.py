"""
script to generate training data for the SVM classifying with training
images in the image_data/ diretcory with a folder for each person.

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
import numpy as np
import os
from edgetpu.basic.basic_engine import BasicEngine
from PIL import Image
import csv


def main(user_id, output_file='training_data.txt'):
    # initial the facenet TFLite model
    engine = BasicEngine("../src/models/facenet_edgetpu.tflite")
    # list of people (subdirectory folder names)
    people = [person for person in os.listdir("image_data/")] if user_id == "-1" else [str(user_id)]
    with open(output_file, 'a+') as f:
        writer = csv.writer(f)
        for person in people:
            image_names = [image for image in os.listdir("image_data/" + person)]
            # run inferece on each mage in the directory
            for image_name in image_names:
                image = Image.open("image_data/" + person + '/' + image_name)
                print("\t->" + person + '/' + image_name)
                # run inference
                engine.RunInference(np.array(image).flatten())
                value = np.zeros(513).astype(object)
                value[0] = str(person).replace('_', ' ')
                value[1:] = engine.get_raw_output()
                # append new label and face embedding pair of the image to the output file
                writer.writerow(value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--user', help='user id, -1 for all')
    parser.add_argument('--output', help='output file file')
    args = parser.parse_args()
    main(args.user, args.output)
