from ultralytics import YOLO
import cv2
import math 
import torch
import time
from person import Person
from hungarian import hungarian
from box import Box
from color import *

##### Hyperparameter######
color_cost_alpha=10
people_id = 0
frame = 0
grace_period = 50
##########################


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# function to draw bounding boxes and display other details on frames
def draw_box(img, box, person, col=(255,0,0), label=None):
    corners = box.get_corners()
    x1, y1, x2, y2 = int(corners[0]), int(corners[1]), int(corners[2]), int(corners[3])

    # put box in cam
    cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
    if not label:
        label = 'confidence: ' + str(box.confidence) + ' ID: ' + str(person.id)
    # object details
    cv2.putText(img, label, [x1, y1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

# webcam
cap = cv2.VideoCapture('videos\\20240430_182615.mp4') # can replace '0' with path to any video you want to run the system on
cap.set(3, 640)
cap.set(4, 480)


# model (make sure you have the .pt file in the same directory as this python file)
model = YOLO("best.pt")
model.to(device)

# list of people that will be tracked by the system
people = []

start_time = time.time # float in seconds

while True:
    #Taking image
    frame += 1
    # debugging print statements
    # print('Number of counted people', people_id)
    # print('\n\n\n\nFRAME', frame)
    # print('PEOPLE')
    # for person in people:
    #     print("PERSON", person.id, "STATES:", person.state, person.bounding_box.position, person.prediction_box.position)
    curr_time = time.time()
    success, img = cap.read()
    results = model(img, stream=True) # run frame through model
    
    bounding_boxes = [] # list of all bounding boxes detected in the frame by the model

    # coordinates
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # add bounding box to list of new bounding boxes detected this frame
            confidence = math.ceil((box.conf[0]*100))/100

            # ignore bounding boxes with low confidence values
            if confidence < 0.49:
                continue
            
            # get bounding box information
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2) # convert to int values
            bounding_boxes.append(Box(frame, confidence, x1=x1, y1=y1, x2=x2, y2=y2))
    
    # debugging print statements
    # print('BOXES')
    # for box in bounding_boxes:
    #     print('box', box.confidence, box.position)
    # print('')

    if len(people) == 0:
        # add all the bounding boxes as new people
        for box in bounding_boxes:
            person = Person(box, curr_time, people_id)
            person.color=ColorFeature(img,box)
            people.append(person)
            draw_box(img, bounding_boxes[-1], people[-1])
            people_id += 1
    else:
        for person in people:
            draw_box(img, person.prediction_box, person, (0, 0, 255), 'prediction for ' + str(person.id))
            person.predict(curr_time)

        # generate cost matrix and run hungarian
        if len(bounding_boxes) > 0:
            cost_matrix = []
            for box in bounding_boxes:
                row = []
                for person in people:
                    row.append(person.prediction_box.euclidean_distance(box))
                cost_matrix.append(row)

            
            color_cost_matrix=color_cost_alpha*ColorDistance(img,bounding_boxes,people)
            # print('Cost matrix of Euclidean Distance',cost_matrix)
            # print('Cost matrix of Color Distance',color_cost_matrix)
            cost_matrix=cost_matrix+color_cost_matrix

            # print('COST MATRIX:\n', cost_matrix)
            assignments = hungarian(cost_matrix)
            # print('ASSIGNMENTS:', assignments)
            
            for i in range(len(assignments)):
                # if a person was not assigned a boundding box, check to see if they were seen recently. If not, mark them for deletion
                if i >= len(bounding_boxes):
                    # print("PERSON", people[assignments[i]].id, "WITH STATES:", person.state, person.bounding_box.position, person.prediction_box.position, 'NOT SEEN... USING PREDICTION')
                    if frame - people[i].frame_history[-1] > grace_period:
                        people[assignments[i]].delete = True # if person has not been seen for 10 frames, delete the person
                    else:
                        # people[assignments[i]].predict(curr_time)
                        draw_box(img, person.prediction_box, person, (0, 0, 255), 'prediction for ' + str(person.id)) # draw red bounding box for predicted Kalman states
                # if the bounding box is not associated to an already-existing person, create a new person for it and draw the bounding box
                elif assignments[i] >= len(people):   
                    people.append(Person(bounding_boxes[i], curr_time, people_id)) # assign bounding box to new person
                    people[-1].color=ColorFeature(img,bounding_boxes[i])
                    people_id += 1 
                    draw_box(img, bounding_boxes[i], people[-1]) # draw blue bounding box for model prediction

                # otherwise the bounding box is associated to a valid person so the bounding box is used to update the person's Kalman state
                else:
                    # print("PERSON", people[assignments[i]].id, "WITH STATES:", people[assignments[i]].state, people[assignments[i]].bounding_box.position, people[assignments[i]].prediction_box.position, 'ASSIGNED TO BOX AT', bounding_boxes[i].position)
                    people[assignments[i]].update(curr_time, bounding_boxes[i])
                    # people[assignments[i]].predict(curr_time) # debug
                    draw_box(img, bounding_boxes[i], people[assignments[i]]) # draw blue bounding box for model prediction
        # if not bounding boxes were detected, automatically update all people and check if they should be deleted
        else:
            for person in people:
                if frame - person.frame_history[-1] > grace_period:
                        person.delete = True # if person has not been seen for 10 frames, delete the person
        
        people = [person for person in people if not person.delete] # delete all people who were marked for deletion
    
    cv2.imshow('Webcam', img)
    
    if cv2.waitKey(1) == ord('q'):
        break
    # time.sleep(1) # for debugging purposes to slow down the system and analyze frames closely

print("A total of", people_id, "pedestrian(s) were seen")
cap.release()
cv2.destroyAllWindows()
