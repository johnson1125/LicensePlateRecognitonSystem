import cv2
import easyocr
from ultralytics import YOLO
from utils.readLicensePlate import readLicensePlate
from utils.utils import visualize

# load models
license_plate_detector = YOLO('models/yolo/best.pt')
def yoloRealTimeDetect() :
    cap = cv2.VideoCapture(1)
    cap.set(3, 640)
    cap.set(4, 480)
    threshold = 0.8
    while True:
        ret, img= cap.read()
        results = license_plate_detector(img,stream=True)

        # coordinates
        for r in results:
            boxes = r.boxes

            for box in boxes:
                if (box.conf[0] > 0.80):
                    # bounding box
                    x1, y1, x2, y2 = 0,0,0,0
                    x1, y1, x2, y2 = box.xyxy[0]
                    license_plateScore=box.conf[0]
                    imgCopy = img.copy()
                    license_plate = readLicensePlate(imgCopy,x1,y1,x2,y2)
                    visualize(img,license_plateScore,license_plate,x1,y1,x2,y2)

        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()