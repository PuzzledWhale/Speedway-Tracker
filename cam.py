from ultralytics import YOLO
import cv2
import math 
import torch
import time
from person import Person
from hungarian import hungarian
from box import Box
from color import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def draw_box(img, box, person, col=(255,0,0), label=None):
    corners = box.get_corners()
    x1, y1, x2, y2 = int(corners[0]), int(corners[1]), int(corners[2]), int(corners[3])

    # put box in cam
    cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
    if not label:
        label = 'confidence: ' + str(box.confidence) + ' ID: ' + str(person.id)
    # object details
    cv2.putText(img, label, [x1, y1], cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

# webcam
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('videos/20240430_182615.mp4')
cap.set(3, 640)
cap.set(4, 480)

people_id = 0

frame = 0
frame_thresh = 1 # set frame threshold to 1 second for now

# model (make sure you have the .pt file in the same directory as this python file)
model = YOLO("best.pt")
model.to(device)

people = []
previous_img = []

grace_period = 50

start_time = time.time # float in seconds
print('starting up camera. Time is:', start_time) 

while True:
    frame += 1
    print('\n\n\n\nFRAME', frame)
    print('PEOPLE')
    for person in people:
        print("PERSON", person.id, "STATES:", person.state, person.bounding_box.position, person.prediction_box.position)
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
            
            # get bounding box information
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2) # convert to int values
            bounding_boxes.append(Box(frame, confidence, x1=x1, y1=y1, x2=x2, y2=y2))
    
    print('BOXES')
    for box in bounding_boxes:
        print('box', box.confidence, box.position)
    print('')
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
        
        # for person in people:
        #     draw_box(img, person.prediction_box, person, (0, 0, 255), 'prediction for ' + str(person.id))

        # generate cost matrix and run hungarian
        if len(bounding_boxes) > 0:
            cost_matrix = []
            for box in bounding_boxes:
                row = []
                for person in people:
                    # row.append(-1 * person.prediction_box.intersect_over_union(box))
                    row.append(person.prediction_box.euclidean_distance(box))
                cost_matrix.append(row)

            previous_boxes=[]
            for person in people :
                previous_boxes.append(person.bounding_box)
            
            # # Cost_matrix for color match
            # if len(previous_img) != 0:
            #     # not first detection 
            #     cost_matrix=cost_matrix+ColorDistance(previous_img,previous_boxes,img,bounding_boxes)
            # previous_img=img # updating previous image

            assignments = hungarian(cost_matrix)
            print('COST MATRIX:\n', cost_matrix)
            print('ASSIGNMENTS:', assignments)
            for i in range(len(assignments)):
                # person not seen, use prediction for update
                if i >= len(bounding_boxes):
                    print("PERSON", people[assignments[i]].id, "WITH STATES:", person.state, person.bounding_box.position, person.prediction_box.position, 'NOT SEEN... USING PREDICTION')
                    # people[assignments[i]].update(None)
                    if frame - people[i].frame_history[-1] > grace_period:
                        people[assignments[i]].delete = True # if person has not been seen for 10 frames, delete the person
                # new frame
                elif assignments[i] >= len(people):   
                    people.append(Person(bounding_boxes[i], curr_time, people_id)) # assign bounding box to new person
                    people_id += 1 
                    draw_box(img, bounding_boxes[i], people[-1])

                # make update with new frame
                else:
                    print("PERSON", people[assignments[i]].id, "WITH STATES:", people[assignments[i]].state, people[assignments[i]].bounding_box.position, people[assignments[i]].prediction_box.position, 'ASSIGNED TO BOX AT', bounding_boxes[i].position)
                    people[assignments[i]].update(bounding_boxes[i])
                    draw_box(img, bounding_boxes[i], people[assignments[i]])
        # if not bounding boxes were detected, automatically update all people and check if they should be deleted
        else:
            for person in people:
                if frame - person.frame_history[-1] > grace_period:
                        person.delete = True # if person has not been seen for 10 frames, delete the person
        
        people = [person for person in people if not person.delete] # delete all people who have not been seen for 10 frames
    
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break
    # time.sleep(2) # for debugging purposes

cap.release()
cv2.destroyAllWindows()
