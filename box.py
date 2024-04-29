import numpy as np

class Box():
    def __init__(self, x1, y1, x2, y2, frame):
        self.position = np.array([((x1 + x2) / 2), ((y1 + y2) / 2)]) # store bounding box center position
        self.size = [x2-x1, y2-y1]
        self.frame = frame

    def get_corners(self):
        half_width = self.size[0] / 2
        half_height = self.size[1] / 2
        top_left = np.array(self.position[0] - half_width, self.position[1] + half_height)
        bot_right = np.array(self.position[0] + half_width, self.position[1] - half_height)
        return top_left, bot_right
    
    # find the intersectiuon over union metric between this box and another box
    def intersect_over_union(box2):
        
