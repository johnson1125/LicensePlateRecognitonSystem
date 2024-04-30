import easyocr
import cv2
from paddleocr import PaddleOCR

reader = easyocr.Reader(['en'], gpu=True)

def readLicensePlate (img,x1,y1,x2,y2) :
    # crop license plate
    license_plate_crop = img[int(y1-3):int(y2+3), int(x1-3): int(x2+3)]

    # process license plate
    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
    license_plate_inverted = cv2.bitwise_not(license_plate_crop_gray)
   # run easyocr
    license_plate_detections = reader.readtext(license_plate_inverted)
    license_plate_text = ''
    for detection in license_plate_detections:
        bbox, texts, score = detection
        texts = texts.upper().replace(' ', '')
        print(texts)
        print(score)

        for text in texts:
            license_plate_text += text

    return license_plate_text
