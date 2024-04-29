from ultralytics import YOLO
import cv2
import math 
import torch
import time
from person import Person
from kalman import KalmanFilter
import hungarian
from box import Box

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

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

            confidence = math.ceil((box.conf[0]*100))/100

            # if confidence is not high enough, continue
            if confidence < 0.4:
                continue

            print("Confidence: ",confidence)
            

            # get bounding box information
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # add bounding box to list of new bounding boxes detected this frame
            bounding_boxes.append(Box(x1, x2, y1, y2, frame))

    
    
    if len(people) == 0:
        # add all the bounding boxes as new people
        for box in bounding_boxes:
            person = Person(box, curr_time)
            # perform color association here! 
    else:
        # run through hungarian algorithm comparing predicted bounding positions
    for person in people:
        # update kalman filter and make new predictions

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()