import cv2 as cv
import mediapipe as mp
import time


mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()


cam = cv.VideoCapture(0)
t1 = 0
t2 = 0
while True:
    success, frame = cam.read()
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame = cv.resize(frame, (1366, 768), interpolation=cv.INTER_AREA)
    results = faceDetection.process(frame_rgb)
    if results.detections:
        for id, detection in enumerate(results.detections):

            bounding_box = detection.location_data.relative_bounding_box
            y, x, z = frame.shape
            rectange_data = int(bounding_box.xmin * x), int(bounding_box.ymin * y), \
                int(bounding_box.width * x), int(bounding_box.height * y)

            cv.rectangle(frame, rectange_data,
                         color=(255, 255, 0), thickness=2)

            cv.putText(frame, f'{int(detection.score[0]*100)}%', (50, 70),
                       cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 0), 2)

    # FPS section starts
    t1 = time.time()
    fps = 1/(t1-t2)
    t2 = t1
    cv.putText(frame, str(int(fps)), (10, 70),
               cv.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 255), 1)
    cv.imshow("video", frame)
    # FPS section ends

    if cv.waitKey(1) & 0xFF == ord('d'):
        break
