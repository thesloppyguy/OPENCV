import cv2 as cv
import mediapipe as mp
import time


class Face_Detection():
    def __init__(self):
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection()

    def findFace(self, frame, draw=True, score=True):
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = cv.resize(frame, (1366, 768), interpolation=cv.INTER_AREA)
        results = self.faceDetection.process(frame_rgb)
        if results.detections:
            for id, detection in enumerate(results.detections):

                bounding_box = detection.location_data.relative_bounding_box
                y, x, z = frame.shape
                rectangle_data = int(bounding_box.xmin * x), int(bounding_box.ymin * y), \
                    int(bounding_box.width * x), int(bounding_box.height * y)
                if draw:
                    frame = self.fancyDraw(frame, rectangle_data)
                    # cv.rectangle(frame, rectangle_data,
                    #              color=(255, 255, 0), thickness=2)
                if score:
                    cv.putText(frame, f'{int(detection.score[0]*100)}%', (50, 70),
                               cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 0), 2)
                return frame, bounding_box

    def fancyDraw(self, img, bbox, l=30, t=5, rt=1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        cv.rectangle(img, bbox, (255, 0, 255), rt)
        # Top Left  x,y
        cv.line(img, (x, y), (x + l, y), (255, 0, 255), t)
        cv.line(img, (x, y), (x, y+l), (255, 0, 255), t)
        # Top Right  x1,y
        cv.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv.line(img, (x1, y), (x1, y+l), (255, 0, 255), t)
        # Bottom Left  x,y1
        cv.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)
        # Bottom Right  x1,y1
        cv.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)
        return img


def main():
    cam = cv.VideoCapture(0)
    fd = Face_Detection()
    t1 = 0
    t2 = 0
    while True:
        success, frame = cam.read()
        frame, bbox = fd.findFace(frame)
        # FPS section starts
        t1 = time.time()
        fps = 1/(t1-t2)
        t2 = t1
        cv.putText(frame, str(int(fps)), (10, 70),
                   cv.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 255), 1)
        # FPS section ends

        cv.imshow("video", frame)

        if cv.waitKey(1) & 0xFF == ord('d'):
            break


if __name__ == '__main__':
    main()
