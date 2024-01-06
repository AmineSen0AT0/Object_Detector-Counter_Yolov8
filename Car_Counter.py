from ultralytics import YOLO
import cv2
import cvzone
import numpy as np
import math
import time
from sort import *


capture = cv2.VideoCapture("Testing Data/Test.mp4") # for video in Testing data
#capture = cv2.VideoCapture(0) # for webcam
capture.set(3 , 1280)
capture.set(4 , 720)

totalVhiclesCounter= set()
model = YOLO('yolov8n.pt')

# TODO: Replace the Total Vhicles Counter by speed Detector , it's simple not complicated at all

# Tracking
tracker = Sort(max_age=20, min_hits= 3, iou_threshold= 0.3)

lineBounding1 = [330, 250, 480, 250]
lineBounding2 = [100, 500, 480, 500]

# Ces classes from Coco Dataset
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

while True:
    Ok, img= capture.read()
    #imgGraphics=cv2.imread("../1- Running Yolo/Images/CAR GRAPHICS.png", cv2.IMREAD_UNCHANGED)
    #cvzone.overlayPNG(img, imgGraphics, (0,0))

    results = model(img, stream=True)
    detections = np.empty((0, 5))

    for r in results:
        boxes= r.boxes
        for box in boxes:
            x1,y1,x2,y2= box.xyxy[0]
            x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2) # ceci for opencv
            #print(x1,y1,x2,y2)
            #cv2.rectangle(img, (x1,y1),(x2,y2) , (0,200, 0), 3)
            w, h =x2-x1 , y2-y1
            # cvzone.cornerRect(img, (x1,y1,w,h) , l=10)
            # Confidence
            confidence = math.floor(box.conf[0]*100)/100
            # print(confidence)
            # Class Name
            detectionClass = int(box.cls[0])
            currentClass=classNames[detectionClass]
            # Not all class is our interest, so we will filter based on IOU & and class Name
            if currentClass== "car" or currentClass=="truck" or currentClass=="bus" or currentClass=="motorbike" and confidence  >0.3:
                cvzone.cornerRect(img, (x1, y1, w, h), l=10 , rt=5)
                cvzone.putTextRect(img , f'{classNames[detectionClass]} {confidence}' , (max(0,x1),max(35,y1)) , scale=1 , thickness=1)
                currentArray = np.array([x1, y1, x2, y2, confidence])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    cv2.line(img,(lineBounding1[0], lineBounding1[1]), (lineBounding1[2], lineBounding1[3]), (255,0,0), 5)
    cv2.line(img,(lineBounding2[0], lineBounding2[1]), (lineBounding2[2], lineBounding2[3]), (0,0,255), 5)


    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # this for opencv
        w, h = x2 - x1, y2 - y1
        print(result)

        # this two lines for printing the id in the bounding boxes, in up we write the same thing but for confidence and class name
        #cvzone.cornerRect(img, (x1, y1, w, h), l=10, rt=2, colorR=(255,0,0))
        #cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

        centerX, centerY = x1 +w//2 , y1+h//2
        cv2.circle(img,(centerX,centerY), 5, (255,0,255), cv2.FILLED)

        if lineBounding1[0]< centerX < lineBounding1[2] and lineBounding1[1]-20 < centerY < lineBounding1[1] + 20:
            totalVhiclesCounter.add(id)
            #cv2.line(img, (lineBounding1[0], lineBounding1[1]), (lineBounding1[2], lineBounding1[3]), (0, 255, 0), 5)

    cvzone.putTextRect(img, f'Count: {len(totalVhiclesCounter)}', (50,50))
    #cv2.putText(img,str(len(totalVhiclesCounter)),(230,90), cv2.FONT_HERSHEY_PLAIN, 5, (100,100,100), 5)
    cv2. imshow("Image" , img)
    cv2.waitKey(1)
