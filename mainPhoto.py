from ultralytics import YOLO
import cv2
import util
from sort.sort import *
from util import get_car, read_license_plate, write_csv
import string
import easyocr





# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}



# load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('./best.pt')

# detect license plate

# load image
img = cv2.imread('./photo10.jpg')

#detect car
detections = coco_model(img)[0]

#detect license plate
license_plates = license_plate_detector(img)[0]

for detection in detections.boxes.data.tolist():
    xcar1, ycar1, xcar2, ycar2, score, class_id = detection


for license_plate in license_plates.boxes.data.tolist():
    x1, y1, x2, y2, license_plateScore, license_plateClass_id = license_plate


color = (0, 255, 0)  # Green color
thickness = 2

# Draw the rectangle on the image
cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), color, thickness)

# Display the image
#cv2.imshow('Image', img)


 # crop license plate
license_plate_crop = img[int(y1):int(y2), int(x1): int(x2)]

# process license plate
license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
license_plate_inverted = cv2.bitwise_not(license_plate_crop_gray)
_, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

cv2.imshow('Image', license_plate_inverted)

# read license plate number
license_plate_detections = reader.readtext(license_plate_inverted)

license_plate_text =''

for detection in license_plate_detections:
    bbox, texts, score = detection

    texts = texts.upper().replace(' ', '')

    print(texts)
    print(score)
    
    for text in texts:
        license_plate_text += text

    # license_plate_text = util.format_license(text)

print(license_plate_text)


print('hello')

# Wait for any key to close the window
cv2.waitKey(0)
cv2.destroyAllWindows()

