"""
Object Tracker
Some part are modified version of the code from
Simple object tracking with OpenCV by Adrian Rosebrock
https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/

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
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
 
class TrackableItem:
    """
    TrackableItem initialization
    input:
        object_id        'int'       object identification number
        centroid        'list'      a x,y coordinate point of the center of the object
    output:
        None
    """
    def __init__(self, object_id, centroid, rect):
        # store the object ID, then initialize a list of centroids
        # using the current centroid
        self.object_id = object_id
        # center position of the objects
        self.centroids = [centroid]
        # size of the objects
        self.sizes = [(rect[2]-rect[0])*(rect[3]-rect[1])]
        # initialize a boolean used to indicate if the object has
        # already been placed in cart or not
        self.incart = False


    """
    TrackableItem to string
    input:
        None
    output:
        TrackableItem as string
    """
    def __str__(self):
        return "id="+str(self.object_id)+" incart="+str(self.incart)+" centroids="+str(self.centroids)

    # add new objec size to list from bounding box rect
    def append_size(self, rect):
        self.sizes.append((rect[2]-rect[0])*(rect[3]-rect[1]))


class ObjectTracker():
    """
    ObjectTracker initialization
    input:
        disappeared_threshold  'int'       maximium number of frame as disappeared (Default: 50)
    output:
        None
    """
    def __init__(self, disappeared_threshold=30, using_iou=True):
        # initialize the next unique object ID along with three ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid, bounding box, and number of consecutive 
        # frames it has been marked as "disappeared", respectively
        self.next_object_id = 0
        self.centroids = OrderedDict()
        self.disappeared = OrderedDict()
        self.object_rects = OrderedDict()
 
        # store the number of maximum consecutive frames loss allowed
        # before deregister the object from tracking
        self.disappeared_threshold = disappeared_threshold
        # flag for using iou trcking instead of centroid
        self.using_iou = using_iou


    def __str__(self):
        return str(self.centroids)


    """
    clear store data
    input:
        None
    output:
        None
    """
    def clear(self):
        self.next_object_id = 0
        self.centroids.clear()
        self.disappeared.clear()
        self.object_rects.clear()


    """
    add a new object
    input:
        centroid        'list'      a x,y coordinate point of the center of the new object
    output:
        None
    """
    def register(self, centroid, rect):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.centroids[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.object_rects[self.next_object_id] = rect
        self.next_object_id += 1


    """
    remove an object with object_id
    input:
        object_id        'int'       object identification number of the object to be removed
    output:
        None
    """
    def deregister(self, object_id):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.centroids[object_id]
        del self.disappeared[object_id]
        del self.object_rects[object_id]


    """
    update the tracking objects based on the closest euclidean distance
    input:
        rects           'list'       list of detected object bounding box coordinates
    output:
        objects         'OrderedDict' contain all the tacking objects
    """
    def update(self, rects):
        # list of objects are deregister in this frame
        disappeared_object_ids = []
        # check to see if the list of input bounding box rectangles is empty
        if len(rects) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
 
                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if self.disappeared[object_id] > self.disappeared_threshold:
                    self.deregister(object_id)
                    disappeared_object_ids.append(object_id)
 
            # return early as there are no info to update
            return self.centroids, self.object_rects, disappeared_object_ids

        # initialize an array of input centroids for the current frame
        input_centroids = np.zeros((len(rects), 2), dtype="int")

        # loop over the bounding box rectangles
        for (i, (left, top, right, bottom)) in enumerate(rects):
            # use the bounding box coordinates to derive the centroid
            c_x = int((left + right) / 2.0)
            c_y = int((top + bottom) / 2.0)
            input_centroids[i] = (c_x, c_y)

        if len(rects) > 1:
            # remove any duplicates that are very close or completely overlapped
            input_centroids, rects = self.remove_duplicates(input_centroids, rects)

        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.centroids) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i], rects[i])

        # otherwise, are are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            object_ids = list(self.centroids.keys())
            objectCentroids = list(self.centroids.values())
            objectRects = list(self.object_rects.values())
            
            if self.using_iou: 
                # compute the ratio of intersection over union between the
                # the bounding box of each pair of object and input centroids
                corr_matrix = self.compute_iou(np.array(objectRects), np.array(rects))
                # Find the largest value in each row and then sort the row
                # indexes based on their maximium values
                rows = corr_matrix.max(axis=1).argsort()[::-1]
                # finding the largest value in each column and then
                # sorting using the previously computed row index list
                cols = corr_matrix.argmax(axis=1)[rows]
            else:
                # compute the distance between each pair of object centroids 
                # and input centroids
                corr_matrix = dist.cdist(np.array(objectCentroids), input_centroids)
                # Find the smallest value in each row and then sort the row
                # indexes based on their minimum values
                rows = corr_matrix.min(axis=1).argsort()
                # finding the smallest value in each column and then
                # sorting using the previously computed row index list
                cols = corr_matrix.argmin(axis=1)[rows]

            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            used_rows = set()
            used_cols = set()
            # loop over the combination of the (row, column) index tuples
            for (row, col) in zip(rows, cols):
                # skip used row and columns
                if row in used_rows or col in used_cols:
                    continue
                # Get the object ID for the current row, set its new 
                # info, and reset the disappeared counter
                object_id = object_ids[row]
                self.centroids[object_id] = input_centroids[col]
                self.object_rects[object_id] = rects[col]
                self.disappeared[object_id] = 0
                # mark the row, col as used
                used_rows.add(row)
                used_cols.add(col)
            # compute both the row and column index we have NOT yet
            # examined
            unused_rows = set(range(0, corr_matrix.shape[0])).difference(used_rows)
            unused_cols = set(range(0, corr_matrix.shape[1])).difference(used_cols)

            # If the number of tracked objects is equal or greater
            # than the number of detected objects, then need to check 
            # and see if some of these objects have potentially disappeared
            if corr_matrix.shape[0] >= corr_matrix.shape[1]:
                # loop over the unused row indexes
                for row in unused_rows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
 
                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.disappeared[object_id] > self.disappeared_threshold:
                        self.deregister(object_id)
                        disappeared_object_ids.append(object_id)
            # else register each new object as a trackable object
            else:
                for col in unused_cols:
                    self.register(input_centroids[col], rects[col])
        # return the set of trackable objects
        return self.centroids, self.object_rects, disappeared_object_ids


    # remove any duplicates rects detected by euclidean distance of centriod,
    # and the amount of overlapping
    def remove_duplicates(self, centroids, rects):
        D = dist.cdist(centroids, centroids)
        duplicate_index = []
        for i in range(D.shape[0]):
            for j in range(i+1, D.shape[1]):
                index = self.check_overlap(rects[i], rects[j])
                if D[i][j] <= 4:
                    duplicate_index.append(j)
                elif index != 0:
                    duplicate_index.append(i if index==1 else j)

        print("rects:",rects)
        print("duplicate_index:",duplicate_index)
        return np.delete(centroids, duplicate_index, axis=0), np.delete(rects, duplicate_index, axis=0)
    

    # check if two rects are overlapping by checking intersect over area ratio 
    def check_overlap(self, rect_a, rect_b):
        a_x = max(rect_a[0], rect_b[0])
        a_y = max(rect_a[1], rect_b[1])
        b_x = min(rect_a[2], rect_b[2])
        b_y = min(rect_a[3], rect_b[3])
     
        # compute the area of intersection rectangle
        intersect_area = max(0, b_x - a_x + 1) * max(0, b_y - a_y + 1)

        # no intersection
        if intersect_area == 0:
            return 0
        # compute intersect area over area of rect a
        rect_a_area = (rect_a[2] - rect_a[0] + 1) * (rect_a[3] - rect_a[1] + 1)
        if  rect_a_area*0.9 <= intersect_area <= rect_a_area*1.1:
            return 1
        # compute intersect area over area of rect b
        rect_b_area = (rect_b[2] - rect_b[0] + 1) * (rect_b[3] - rect_b[1] + 1)
        if  rect_b_area*0.9 <= intersect_area <= rect_b_area*1.1:
            return 2
        return 0


    # Compute the iou of each objects pairs from the two lists
    # Input: two narray of bbox
    def compute_iou(self, rects_a, rects_b):
        det = np.zeros(shape=(rects_a.shape[0], rects_b.shape[0]), dtype="float16")
        for i in range(det.shape[0]):
            for j in range(det.shape[1]):
                det[i,j] = self.iou(rects_a[i], rects_b[j])
        return det


    # intersection_over_union of two bounding box
    def iou(self, rect_a, rect_b):
        # determine the (x, y)-coordinates of the intersection rectangle
        a_x = max(rect_a[0], rect_b[0])
        a_y = max(rect_a[1], rect_b[1])
        b_x = min(rect_a[2], rect_b[2])
        b_y = min(rect_a[3], rect_b[3])
     
        # compute the area of intersection rectangle
        intersect_area = max(0, b_x - a_x + 1) * max(0, b_y - a_y + 1)

        # no intersection return zero
        if intersect_area == 0:
            return 0
     
        # compute the area of both the both rects
        rect_a_area = (rect_a[2] - rect_a[0] + 1) * (rect_a[3] - rect_a[1] + 1)
        rect_b_area = (rect_b[2] - rect_b[0] + 1) * (rect_b[3] - rect_b[1] + 1)
     
        # compute the intersection over union
        iou = intersect_area / float(rect_a_area + rect_b_area - intersect_area)
        # return the intersection over union value
        return iou