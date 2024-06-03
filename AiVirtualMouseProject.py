import cv2
import numpy as np
import HandTrackingModule as htm
import time
import pyautogui

##########################
wCam, hCam = 640, 480
frameR = 100  # Frame Reduction
smoothening = 7
dotSize = 5  # Radius of the dot
clickDotSize = 10  # Radius of the dot for click actions
dragDistance = 40  # Distance threshold for drag and drop
#########################

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0
dragging = False

cap = cv2.VideoCapture(0)  # Try different indices if necessary
cap.set(3, wCam)
cap.set(4, hCam)

if not cap.isOpened():
    print("Error: Could not open video capture")
    exit()

detector = htm.handDetector(maxHands=1)
wScr, hScr = pyautogui.size()  # Get the size of the primary monitor

while True:
    # 1. Find hand Landmarks
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    # 2. Get the tip of the index and middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]  # Index finger tip
        x2, y2 = lmList[12][1:]  # Middle finger tip

        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

        # 4. Move the mouse when both index and middle fingers are up and distance is greater than dragDistance
        if fingers == [0, 1, 1, 0, 0]:
            # Calculate the distance between the index and middle fingers
            length, img, _ = detector.findDistance(8, 12, img)

            # 5. Convert Coordinates
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            # 6. Smoothen Values
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # 7. Move Mouse
            pyautogui.moveTo(wScr - clocX, clocY)
            cv2.circle(img, (x1, y1), dotSize, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

            # 8. Check if the distance is below the dragDistance threshold
            if length < dragDistance and not dragging:
                pyautogui.mouseDown()
                dragging = True
                cv2.circle(img, (x1, y1), clickDotSize, (0, 255, 0), cv2.FILLED)
            elif length >= dragDistance and dragging:
                pyautogui.mouseUp()
                dragging = False
                cv2.circle(img, (x1, y1), clickDotSize, (0, 0, 255), cv2.FILLED)

        # 9. Left click when index finger is down and middle finger is up
        elif fingers == [0, 0, 1, 0, 0]:
            pyautogui.click(button='left')
            cv2.circle(img, (x2, y2), clickDotSize, (0, 255, 0), cv2.FILLED)

        # 10. Right click when index finger is up and middle finger is down
        elif fingers == [0, 1, 0, 0, 0]:
            pyautogui.click(button='right')
            cv2.circle(img, (x1, y1), clickDotSize, (0, 0, 255), cv2.FILLED)

    # 11. Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # 12. Display
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
