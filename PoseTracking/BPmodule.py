import cv2 as cv
import mediapipe as mp
import time


class PoseDetection():

    def __init__(self, mode=False, upBody=False, smooth=True,
                 detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(self.mode, self.upBody, self.smooth,
                                      self.detectionCon, self.trackCon)

    def findBody(self, img, draw=True):
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = self.mp_pose.process(img)
        if results.pose_landmarks:
            if draw:
                self.mp_drawing.draw_landmarks(
                    img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

    def findPoints(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                x, y = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, x, y])
                if draw:
                    cv.circle(img, (x, y), 5, (255, 0, 0), cv.FILLED)
        return self.lmList


def main():
    t1 = 0
    t2 = 0
    video = cv.VideoCapture("./DATA/v1.mp4")
    detector = PoseDetection()
    while True:
        success, frame = video.read()
        frame = cv.resize(frame, (1366, 768), interpolation=cv.INTER_AREA)
        detector.findBody(frame)
        list = detector.findPoints(frame)
        cv.imshow("video", frame)

        # fps counter
        t1 = time.time()
        fps = 1/(t1-t2)
        t2 = t1
        cv.putText(frame, str(int(fps)), (10, 70),
                   cv.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 255), 1)
        cv.imshow("video", frame)
        # fps counter ends

        if cv.waitKey(1) & 0xFF == ord('d'):
            break


if __name__ == '__main__':
    main()
