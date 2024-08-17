import cv2
import os
from cvzone.HandTrackingModule import HandDetector
import numpy as np

# Set up camera and presentation dimensions
width, height = 1280, 720
folderPath = "Presentation"

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# Load presentation images
pathImages = sorted(os.listdir(folderPath), key=len)
imgNumber = 0
gestureThreshold = 380
buttonPressed = False
buttonCounter = 0
buttonDelay = 30

# Set dimensions for the camera feed within the presentation
camera_feed_width, camera_feed_height = 500, 160  # Double the width

# Hand detector setup
detector = HandDetector(detectionCon=0.8, maxHands=1)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
    imgCurrent = cv2.imread(pathFullImage)

    # Detect hands and annotate the camera feed
    hands, img = detector.findHands(img)
    cv2.line(img, (0, gestureThreshold), (width, gestureThreshold), (0, 255, 0), 10)

    if hands and not buttonPressed:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        cx, cy = hand['center']
        lmList = hand['lmList']

        # Correct interpolation to move pointer across full slide
        xVal = int(np.interp(lmList[8][0], [width // 2, width], [0, width]))
        yVal = int(np.interp(lmList[8][1], [150, height - 150], [0, height]))
        indexFinger = xVal, yVal

        if cy <= gestureThreshold:
            if fingers == [1, 0, 0, 0, 0]:
                if imgNumber > 0:
                    buttonPressed = True
                    imgNumber -= 1

            if fingers == [0, 0, 0, 0, 1]:
                if imgNumber < len(pathImages) - 1:
                    buttonPressed = True
                    imgNumber += 1

        if fingers == [0, 1, 1, 0, 0]:
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)

        if fingers == [0, 1, 0, 0, 1]:
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)

    if buttonPressed:
        buttonCounter += 1
        if buttonCounter > buttonDelay:
            buttonCounter = 0
            buttonPressed = False


    # Resize the camera feed to fit in the top right corner
    imgSmall = cv2.resize(img, (camera_feed_width, camera_feed_height))

    # Place the resized camera feed on top right corner of the presentation
    h, w, _ = imgCurrent.shape
    imgCurrent[0:camera_feed_height, w - camera_feed_width:w] = imgSmall

    # Display the combined presentation and camera feed
    cv2.imshow("Slides", imgCurrent)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
