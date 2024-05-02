import numpy as np
import cv2
from tensorflow.lite.python.interpreter import Interpreter
from utils.readLicensePlate import readLicensePlate
from utils.utils import visualize
from sort.sort import *


# setup
modelpath = 'detect.tflite'
min_conf = 0.50
mot_tracker = Sort()
interpreter = Interpreter(model_path=modelpath)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
float_input = (input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5

def ssdRealTimeDetect():
    global license_plateScore
    pointx1 = 0.0
    pointx2 = 0.0
    pointy1 = 0.0
    pointy2 = 0.0
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    carPlate_dict = {}
    while (True):
        ret, frame = cap.read()
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imH, imW, _ = frame.shape
        image_resized = cv2.resize(image_rgb, (width, height))
        input_data = np.expand_dims(image_resized, axis=0)
        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if float_input:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        detections_ = []
        boxes = interpreter.get_tensor(output_details[1]['index'])[0]  # Bounding box coordinates of detected objects
        scores = interpreter.get_tensor(output_details[0]['index'])[0]  # Confidence of detected objects

        for i in range(len(scores)):
            if ((scores[i] > min_conf) and (scores[i] <= 1.0)):
                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                license_plateScore = scores[i]
                ymin = int(max(1, (boxes[i][0] * imH)))
                xmin = int(max(1, (boxes[i][1] * imW)))
                ymax = int(min(imH, (boxes[i][2] * imH)))
                xmax = int(min(imW, (boxes[i][3] * imW)))
                detections_.append([xmin, ymin, xmax, ymax, license_plateScore])

        print(len(detections_))
        detections_np = np.array(detections_).reshape(-1, 5)
        detections_bboxes = detections_np[:, :4]
        track_ids = mot_tracker.update(np.asarray(detections_bboxes))
        print(len(track_ids))
        if track_ids.shape[0] > 0 and track_ids.shape[1] == 5:
            for track_id in track_ids:
                pointx1, pointy1, pointx2, pointy2, carPlate_id = track_id
                framecpy = frame.copy()
                license_plate_result = readLicensePlate(framecpy, pointx1, pointy1, pointx2, pointy2)
                if carPlate_id in carPlate_dict:
                    print(carPlate_id)
                    print(carPlate_dict[carPlate_id])

                    dict_license_plate_data = carPlate_dict[carPlate_id][0]
                    print(dict_license_plate_data)
                    dict_license_plate, dict_license_plate_score = dict_license_plate_data
                    visualize(frame, license_plateScore, dict_license_plate, pointx1, pointy1, pointx2,
                              pointy2)
                    # if dict_license_plate_score < license_plate_result[1]:
                    #     carPlate_dict.update({carPlate_id: [(license_plate_result[0],license_plate_result[1])]})
                    #     visualize(img, license_plateScore, dict_license_plate, pointx1, pointy1, pointx2, pointy2)
                    # else:

                else:
                    carPlate_dict[carPlate_id] = [(license_plate_result[0], license_plate_result[1])]
                    visualize(frame, license_plateScore, license_plate_result[0], pointx1, pointy1, pointx2, pointy2)

                # license_plate = readLicensePlate(framecpy, xmin, ymin, xmax, ymax)
                # # Draw label
                # visualize(frame, license_plateScore, license_plate, xmin, ymin, xmax, ymax)



        cv2.imshow('RealTime license Plate System', frame)
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()
ssdRealTimeDetect()