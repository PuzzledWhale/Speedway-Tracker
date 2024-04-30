from ultralytics import YOLO
import cv2
import math 
import torch
import time
from person import Person
import hungarian
from box import Box
from color import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def draw_box(img, box, person):
    corners = box.get_corners()
    x1, y1, x2, y2 = corners[0], corners[1], corners[2], corners[3]

    # put box in cam
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

    # object details
    txt = 'confidence: ' + str(box.confidence) + ' ID: ' + person.id, 
    org = [x1, y1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2
    cv2.putText(img, txt, org, font, fontScale, color, thickness)

# webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

people_id = 0

frame = 0
frame_thresh = 1 # set frame threshold to 1 second for now

# model (make sure you have the .pt file in the same directory as this python file)
model = YOLO("best.pt")
model.to(device)

people = []
start_time = time.time # float in seconds
print('starting up camera. Time is:', start_time) 

while True:
    frame += 1
    curr_time = time.time()
    success, img = cap.read()
    results = model(img, stream=True)
    
    bounding_boxes = []
    # coordinates
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # add bounding box to list of new bounding boxes detected this frame
            confidence = math.ceil((box.conf[0]*100))/100

            # if confidence is not high enough, continue
            if confidence < 0.4:
                continue

            print("Confidence: ",confidence)
            
            # get bounding box information
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
            bounding_boxes.append(Box(frame, confidence, x1=x1, y1=y1, x2=x2, y2=y2))
    
    
    if len(people) == 0:
        # add all the bounding boxes as new people
        for box in bounding_boxes:
            person = Person(box, curr_time, people_id)
            people_id += 1
            people.append(person)
    else:
        #  predict bounding box positions at new time and assign through hungarian algorithm
        for person in people:
            person.predict(curr_time)
        
        # generate cost matrix and run hungarian
        cost_matrix = []
        for person in people:
            row = []
            for box in bounding_boxes:
                # row.append(1.0 / person.prediction_box.intersection_over_union(box))
                row.append(person.prediction_box.euclidean_distance(box))

        previous_boxes=[]
        for person in people :
            previous_boxes.append(person.bounding_box)
        
        # Cost_matrix for color match
        if len(previous_img) != 0:
            # not first detection 
            cost_matrix=cost_matrix+ColorDistance(previous_img,previous_boxes,img,bounding_boxes)
        previous_img=img # updating previous image

        assignments = hungarian(cost_matrix)

        for i in range(assignments):
            # person not seen, use prediction for update
            if i >= len(bounding_boxes):
                people[assignments[i]].update(None)
                if frame - people[i].frame_history[-1] > 10:
                    people[assignments[i]].delete = True # if person has not been seen for 10 frames, delete the person
            # new frame
            elif assignments[i] >= len(people):   
                people.append(Person(bounding_boxes[i], curr_time, people_id)) # assign bounding box to new person
                people_id += 1 
                draw_box(img, bounding_boxes[i], people[-1])

            # make update with new frame
            else:
                people[assignments[i]].update(bounding_boxes[i])
                draw_box(img, bounding_boxes[i], people[assignments[i]])
            
        
        people = [person for person in people if not person.delete] # delete all people who have not been seen for 10 frames
    
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
