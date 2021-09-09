import cv2 as cv
import mediapipe as mp
import time


cam = cv.VideoCapture(0)
t1 = 0
t2 = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)


while True:
    success, frame = cam.read()
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame = cv.resize(frame, (1366, 768), interpolation=cv.INTER_AREA)
    results = faceMesh.process(frame_rgb)
    if results.multi_face_landmarks:
        for item in results.multi_face_landmarks:
            mpDraw.draw_landmarks(
                frame, item, mpFaceMesh.FACEMESH_TESSELATION, drawSpec, drawSpec)
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
