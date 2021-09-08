import cv2 as cv
import mediapipe as mp
import time
cam = cv.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

t1 = 0
t2 = 0

while True:
    success, frame = cam.read()
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for item in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, item, mp_hands.HAND_CONNECTIONS)

    t1 = time.time()
    fps = 1/(t1-t2)
    t2 = t1
    cv.putText(frame, str(int(fps)), (10, 70),
               cv.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 255), 1)
    cv.imshow("video", frame)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break


cam.release()
cv.destroyAllWindows()

cv.waitKey(0)
