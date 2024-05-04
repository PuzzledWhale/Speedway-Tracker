import numpy as np
import math

class Box():
    # if x2 and y2 are provided then find center between point1 and point2. Otherwise assume point1 is the center
    def __init__(self, frame, confidence, x1, y1, x2 = -1, y2 = -1, size=np.array([0,0])):
        self.frame = frame
        self.confidence = confidence
        if x2 == -1 and y2 == -1:
            self.position = np.array([x1, y1])
            self.size = size
        else:
            self.position = np.array([((x1 + x2) / 2), ((y1 + y2) / 2)]) # store bounding box center position
            self.size = np.array([x2-x1, y2-y1])
           


    # get the corners of the bounding box as a numpy array of (x1, y1, x2, y2) where (x1, y1) is the top left corner and (x2, y2) is the bottom right corner
    def get_corners(self):
        half_width = self.size[0] / 2
        half_height = self.size[1] / 2
        return np.array([self.position[0] - half_width, self.position[1] + half_height, self.position[0] + half_width, self.position[1] - half_height])
    
    # find the intersectiuon over union metric between this box (box1) and another box (box2)
    def intersect_over_union(self, box2):
        corners1 = self.get_corners()
        corners2 = box2.get_corners()

        area1 = self.get_area()
        area2 = box2.get_area()

        # find corners of intersection between box1 and box2
        x1, y1, x2, y2 = max(corners1[0], corners2[0]), max(corners1[1], corners2[1]), min(corners1[2], corners2[2]), min(corners1[3], corners2[3])
        intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1+ 1)

        # intersection over union is (area of overlap / area of union)
        return intersection_area / (area1 + area2 - intersection_area) 
    
    # find euclidean distance between this bounding box (box1) and another (box2)
    def euclidean_distance(self, box2):
        return math.sqrt((self.position[0] - box2.position[0])**2 + (self.position[1] - box2.position[1])**2)
    
    # returns the total area of the bouinding box (W * H)
    def get_area(self):
        return self.size[0] * self.size[1]
