import tensorflow as tf
import numpy as np
import cv2

def load_model(model_path):
    return tf.saved_model.load(model_path)

def prepare_input(image_path):
    # Load image and preprocess it
    image = cv2.imread(image_path)
    image = cv2.resize(image, (640, 640)) # Example size, adjust to your model's input
    image = np.expand_dims(image, axis=0) # Add batch dimension
    return tf.convert_to_tensor(image, dtype=tf.uint8)

def run_inference(model, input_tensor):
    return model(input_tensor)
threshold = 0.20

# Example usage
model_path = 'new_model/content/export/saved_model'
model = load_model(model_path)
imgPath = 'photo9.jpg'
input_tensor = prepare_input(imgPath)
img = cv2.imread(imgPath)
detections = run_inference(model, input_tensor)

# Example for extracting information from the detections
detection_boxes = detections['detection_boxes'].numpy().squeeze()
detection_scores = detections['detection_scores'].numpy().squeeze()
detection_classes = detections['detection_classes'].numpy().squeeze()

imH, imW, _ = img.shape

# Draw bounding boxes and labels on the image
for i in range(len(detection_scores)):
    if detection_scores[i] >= threshold:  # Only consider detections above a certain score
        box = detection_boxes[i]
        class_label = detection_classes[i]
        label = '%s: %d%%' % ("License Plate", int(detection_scores[i] * 100))  # Example: 'person: 72%'
        ymin = int(max(1, (detection_boxes[i][0] * imH)))
        xmin = int(max(1, (detection_boxes[i][1] * imW)))
        ymax = int(min(imH, (detection_boxes[i][2] * imH)))
        xmax = int(min(imW, (detection_boxes[i][3] * imW)))

        # Draw the bounding box and label
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(img,  label,(xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


cv2.imshow('Image', img)
# Wait for any key to close the window
cv2.waitKey(0)
cv2.destroyAllWindows()

print("hello")



