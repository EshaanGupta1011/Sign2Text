import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

offset = 26
imgSize = 450
counter = 0

dict = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "K": 9,
    "L": 10,
    "M": 11,
    "N": 12,
    "O": 13,
    "P": 14,
    "Q": 15,
    "R": 16,
    "S": 17,
    "T": 18,
    "U": 19,
    "V": 20,
    "W": 21,
    "X": 22,
    "Y": 23,
}

folder_var = "A"  # Enter alphabet
folder = f"Data/{folder_var}"
name = dict[folder_var]

cap = cv2.VideoCapture(0)  # cap is used to capture the image through camera
detector = HandDetector(maxHands=1)  # an obj of HandDetector used to define max hands to detect

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    try:
        if hands:
            hand = hands[0]
            x, y, w, h = hand["bbox"]

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset: y + h + 25, x - offset: x + w + 50]

            imgCropShape = imgCrop.shape

            aspectRatio = h / w
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))  # (dimension, image)

                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap: wCal + wGap] = imgResize

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))  # (dimension, image)

                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap: hCal + hGap, :] = imgResize

            cv2.imshow("Image Crop", imgCrop)
            cv2.imshow("Image White", imgWhite)

        else:
            cv2.imshow("Image Crop", imgCrop)
            cv2.imshow("Image White", imgWhite)

    except:
        pass

    # Below lines are used to display the live image to the user in a window named image
    cv2.imshow("image", img)
    key = cv2.waitKey(1)

    if key == ord("s"):
        counter += 1
        cv2.imwrite(f"{folder}/{name}_({time.time()}).jpg", imgWhite)  # naming and defining which image to save
        print(counter)
