# Import necessary libraries
import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# Initialize deque arrays to store color points
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]

# Indexes for color points
blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0

# The kernel for dilation purposes
kernel = np.ones((5,5), np.uint8)

# Color options: Blue, Green, Red, Yellow
colors = [(255,0,0), (0,255,0), (0,0,255), (0,255,255)]
colorIndex = 0

# Create a white canvas with color buttons
paintWindow = np.zeros((471,636,3)) + 255
paintWindow = cv2.rectangle(paintWindow, (40,1), (140,65), (0,0,0), 2)
paintWindow = cv2.rectangle(paintWindow, (160,1), (255,65), (255,0,0), 2)
paintWindow = cv2.rectangle(paintWindow, (275,1), (375,65), (0,255,0), 2)
paintWindow = cv2.rectangle(paintWindow, (390,1), (485,65), (0,0,255), 2)
paintWindow = cv2.rectangle(paintWindow, (505,1), (600,65), (0,255,255), 2)

cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,0), 2, cv2.LINE_AA)
cv2.namedWindow("Paint",cv2.WINDOW_AUTOSIZE)

# Initialize mediapipe hand detection
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils  # Utility to draw hand landmarks

# Initialize the webcam
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip the frame horizontally
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Add buttons on frame
    frame = cv2.rectangle(frame, (40, 1), (140, 65), (0, 0, 0), 2)
    frame = cv2.rectangle(frame, (160, 1), (255, 65), (255, 0, 0), 2)
    frame = cv2.rectangle(frame, (275, 1), (375, 65), (0, 255, 0), 2)
    frame = cv2.rectangle(frame, (390, 1), (485, 65), (0, 0, 255), 2)
    frame = cv2.rectangle(frame, (505, 1), (600, 65), (0, 255, 255), 2)

    cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

    # Process the frame through mediapipe hands
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        landmarks = []
        for hand_landmarks in result.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                lmx = int(lm.x * frame.shape[1])
                lmy = int(lm.y * frame.shape[0])
                landmarks.append([lmx, lmy])

            mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)

        fore_finger = (landmarks[8][0], landmarks[8][1])  # Tip of index finger
        thumb = (landmarks[4][0], landmarks[4][1])  # Tip of thumb

        cv2.circle(frame, fore_finger, 3, (0, 255, 0), -1)

        # Check for click gesture (when index finger and thumb are close)
        if abs(thumb[1] - fore_finger[1]) < 30:
            bpoints.append(deque(maxlen=512))
            blue_index += 1
            gpoints.append(deque(maxlen=512))
            green_index += 1
            rpoints.append(deque(maxlen=512))
            red_index += 1
            ypoints.append(deque(maxlen=512))
            yellow_index += 1

        # Check if index finger is in the button region
        elif fore_finger[1] <= 65:
            if 40 <= fore_finger[0] <= 140:
                bpoints = [deque(maxlen=1024)]
                gpoints = [deque(maxlen=1024)]
                rpoints = [deque(maxlen=1024)]
                ypoints = [deque(maxlen=1024)]
                blue_index = green_index = red_index = yellow_index = 0
            elif 160 <= fore_finger[0] <= 255:
                colorIndex = 0  # Blue
            elif 275 <= fore_finger[0] <= 370:
                colorIndex = 1  # Green
            elif 390 <= fore_finger[0] <= 485:
                colorIndex = 2  # Red
            elif 505 <= fore_finger[0] <= 600:
                colorIndex = 3  # Yellow
        else:
            # Add points to the respective deque
            if colorIndex == 0:
                bpoints[blue_index].appendleft(fore_finger)
            elif colorIndex == 1:
                gpoints[green_index].appendleft(fore_finger)
            elif colorIndex == 2:
                rpoints[red_index].appendleft(fore_finger)
            elif colorIndex == 3:
                ypoints[yellow_index].appendleft(fore_finger)

    # Draw lines of all the points
    points = [bpoints, gpoints, rpoints, ypoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

    # Show the final output windows and handle key presses
    cv2.imshow("Output", frame)
    cv2.imshow("Paint", paintWindow)

    # Check if 'q' key is pressed to break the loop
    if cv2.waitKey(1) == ord('q'):
        break

    # Release the webcam and destroy all active windows after the loop
    cap.release()
    cv2.destroyAllWindows()




