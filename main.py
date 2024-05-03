import tkinter as tk
from tkinter import ttk
from tkinter import Label
from tkinter import filedialog
from PIL import Image, ImageTk
from models.yolo.yoloPhoto import yoloDetectPhoto
from models.yolo.yoloRealTimeDetection import yoloRealTimeModelDetect
from models.sdd_MobileNetV2_FpnLite.sddMobileNetV2Photo import ssdDetectPhoto
from models.faster_rcnn.fasterRcnnPhoto import fasterRcnnDetectPhoto
from models.faster_rcnn.fasterRcnnCamera import fasterRcnnRealTimeDetect
from models.sdd_MobileNetV2_FpnLite.ssdMobileNetV2RealTimeDetection import ssdRealTimeModelDetect


def openNewTab(windowTitle,imgPath):
    # Create a new window
    new_window = tk.Toplevel()
    new_window.title(windowTitle)

    # Load the image (replace 'path/to/your/image.jpg' with the actual path)
    img = Image.open(imgPath)
    img = img.resize((640, 640), Image.ANTIALIAS)  # Resize the image if needed
    photo = ImageTk.PhotoImage(img)

def display_error_message(frame, message):
    error_label = tk.Label(frame, text=message, font=("Helvetica", 16), fg="red")
    error_label.place(relx=0.5, rely=0.88, anchor=tk.CENTER)

def yoloPhotoDetection():
    fileTypes = [("Image files", "*.png;*.jpg;*.jpeg")]
    path = tk.filedialog.askopenfilename(filetypes=fileTypes)

    # if file is selected
    if len(path):
        detectionResult = yoloDetectPhoto(path)
        if detectionResult is None:
            display_error_message(ssd_frame, "No car plate detected...")
            return
        img = Image.open(detectionResult[1])
        img = img.resize((460, 460), Image.LANCZOS)  # Resize the image if needed
        photo = ImageTk.PhotoImage(img)

        # Create a label to display the image
        image_label_yolo.config(image=photo)
        image_label_yolo.image = photo  # Keep a reference to prevent garbage collection

        # Create a label to display the detection text
        detection_text_yolo.config(text=f"YOLO: {detectionResult[0]}")

    # if no file is selected, then we are displaying below message
    else:
        print("No file is chosen !! Please choose a file.")

def ssdPhotoDetection():
    fileTypes = [("Image files", "*.png;*.jpg;*.jpeg")]
    path = tk.filedialog.askopenfilename(filetypes=fileTypes)

    # if file is selected
    if len(path):
        detectionResult = ssdDetectPhoto(path)
        if detectionResult is None:
            display_error_message(ssd_frame, "No car plate detected...")
            return
        img = Image.open(detectionResult[1])
        img = img.resize((460, 460), Image.LANCZOS)  # Resize the image if needed
        photo = ImageTk.PhotoImage(img)

        # Create a label to display the image
        image_label_ssd.config(image=photo)
        image_label_ssd.image = photo  # Keep a reference to prevent garbage collection

        # Create a label to display the detection text
        detection_text_ssd.config(text=f"SSD: {detectionResult[0]}")


    # if no file is selected, then we are displaying below message
    else:
        print("No file is chosen !! Please choose a file.")


def fasterRcnnPhotoDetection():
    fileTypes = [("Image files", "*.png;*.jpg;*.jpeg")]
    path = tk.filedialog.askopenfilename(filetypes=fileTypes)

    # if file is selected
    if len(path):
        detectionResult = fasterRcnnDetectPhoto(path)
        if detectionResult is None:
            display_error_message(fasterRcnn_frame, "No car plate detected...")
            return
        img = Image.open(detectionResult[1])
        img = img.resize((460, 460), Image.LANCZOS)  # Resize the image if needed
        photo = ImageTk.PhotoImage(img)

        # Create a label to display the image
        image_label_faster_rcnn.config(image=photo)
        image_label_faster_rcnn.image = photo  # Keep a reference to prevent garbage collection

        # Create a label to display the detection text
        detection_text_faster_rcnn.config(text=f"Faster R-CNN: {detectionResult[0]}")


    # if no file is selected, then we are displaying below message
    else:
        print("No file is chosen !! Please choose a file.")

def yoloRealTimeDetection():
    yoloRealTimeModelDetect()

def ssdRealTimeDetection():
    ssdRealTimeModelDetect()

def fasterRcnnRealTimeDetection():
    fasterRcnnRealTimeDetect()


def on_button_click():
    user_input = entry.get().upper()
    print("User Input:", user_input)

    # Append user input to a text file
    with open("resources/registered_car_plate.txt", "a") as file:
        file.write(user_input + "\n")
        print("User input appended to file.")

    # Clear the text in the entry widget
    entry.delete(0, 'end')

if __name__ == "__main__":
    # defining tkinter object
    app = tk.Tk()
    app.title("Car Plate Number Recognition System")
    app.geometry("1200x900")

    # Adding background color to labels and buttons
    app.option_add("*Label*Background", "white")
    app.option_add("*Button*Background", "lightgreen")

    # Create a notebook (tabbed interface)
    notebook = ttk.Notebook(app)

    # Create frames for each model
    register_frame = ttk.Frame(notebook)
    yolo_frame = ttk.Frame(notebook)
    ssd_frame = ttk.Frame(notebook)
    fasterRcnn_frame = ttk.Frame(notebook)
    yoloLive_frame = ttk.Frame(notebook)
    ssdLive_frame = ttk.Frame(notebook)
    fasterRcnnLive_frame = ttk.Frame(notebook)


    # Add frames to the notebook
    notebook.add(register_frame, text="Register Car Plate")
    notebook.add(yolo_frame, text="YOLO Photo")
    notebook.add(ssd_frame, text="SSD Photo")
    notebook.add(fasterRcnn_frame, text="Faster Rcnn Photo")
    notebook.add(yoloLive_frame, text="YOLO Live")
    notebook.add(ssdLive_frame, text="SSD Live")
    notebook.add(fasterRcnnLive_frame, text="Faster Rcnn Live")


    # Customize tab appearance (increase font size and padding)
    style = ttk.Style()
    style.configure("TNotebook.Tab", font=("Helvetica", 14), padding=[20, 10])

    # Create buttons for each model within their respective frames
    yolo_button1 = tk.Button(yolo_frame, text="Detect Image (YOLO)", command=yoloPhotoDetection, font=("Helvetica", 16),
                             width=25, height=2)  # Adjust width, height, and font size as needed

    ssd_button1 = tk.Button(ssd_frame, text="Detect Image (SSD)", command=ssdPhotoDetection, font=("Helvetica", 16),
                            width=25, height=2)

    fasterRcnn_button1 = tk.Button(fasterRcnn_frame, text="Detect Image (FasterRCNN)", command=fasterRcnnPhotoDetection,
                                   font=("Helvetica", 16), width=25, height=2)

    yolo_button2 = tk.Button(yoloLive_frame, text="Live Detect (YOLO)", command=yoloRealTimeDetection,
                             font=("Helvetica", 16), width=25,
                             height=2)  # Adjust width, height, and font size as needed

    ssd_button2 = tk.Button(ssdLive_frame, text="Live Detect (SSD)", command=ssdRealTimeDetection,
                            font=("Helvetica", 16), width=25, height=2)

    fasterRcnn_button2 = tk.Button(fasterRcnnLive_frame, text="Live Detect (FasterRCNN)",
                                   command=lambda: fasterRcnnRealTimeDetect(fasterRcnnLive_frame),
                                   font=("Helvetica", 16), width=25, height=2)


    # Create a label to display the image for each frame
    image_label_yolo = tk.Label(yolo_frame)
    image_label_yolo.place(relx=0.5, rely=0.55, anchor=tk.CENTER)

    image_label_ssd = tk.Label(ssd_frame)
    image_label_ssd.place(relx=0.5, rely=0.55, anchor=tk.CENTER)

    image_label_faster_rcnn = tk.Label(fasterRcnn_frame)
    image_label_faster_rcnn.place(relx=0.5, rely=0.55, anchor=tk.CENTER)

    # Create a label to display the detection text for each frame
    detection_text_yolo = tk.Label(yolo_frame, font=("Helvetica", 16))
    detection_text_yolo.place(relx=0.5, rely=0.88, anchor=tk.CENTER)

    detection_text_ssd = tk.Label(ssd_frame, font=("Helvetica", 16))
    detection_text_ssd.place(relx=0.5, rely=0.88, anchor=tk.CENTER)

    detection_text_faster_rcnn = tk.Label(fasterRcnn_frame, font=("Helvetica", 16))
    detection_text_faster_rcnn.place(relx=0.5, rely=0.88, anchor=tk.CENTER)

    # Create a label for the text input
    label = tk.Label(register_frame, text="Register Car Plate Number Here", font=("Helvetica", 30))
    label.place(relx=0.5, rely=0.1, anchor=tk.CENTER)

    # Create a label for the text input
    label = tk.Label(register_frame, text="Enter your text:", font=("Helvetica", 16))
    label.place(relx=0.5, rely=0.2, anchor=tk.CENTER)

    # Create a text input field
    entry = tk.Entry(register_frame,  font=("Helvetica", 16))
    entry.place(relx=0.5, rely=0.25, anchor=tk.CENTER)

    # Create a button
    button = tk.Button(register_frame, text="Submit", command=on_button_click,  font=("Helvetica", 16))
    button.place(relx=0.5, rely=0.30, anchor=tk.CENTER)

    # Pack the buttons with customized positions
    yolo_button1.pack(pady=30)
    yolo_button2.pack(pady=30)

    ssd_button1.pack(pady=30)
    ssd_button2.pack(pady=30)

    fasterRcnn_button1.pack(pady=30)
    fasterRcnn_button2.pack(pady=30)

    # Pack the notebook
    notebook.pack(fill="both", expand=True)

    app.mainloop()

