import tkinter as tk
from tkinter import ttk
from tkinter import Label
from tkinter import filedialog
from PIL import Image, ImageTk
from models.yolo.yoloPhoto import yoloDetectPhoto
from models.yolo.yoloCamera import yoloRealTimeDetect
from models.sdd_MobileNetV2_FpnLite.sddMobileNetV2Photo import ssdDetectPhoto
from models.sdd_MobileNetV2_FpnLite.sddMobileNetV2Camera import ssdRealTimeDetect
from models.detectron.fasterRcnnPhoto import fasterRcnnDetectPhoto
from models.detectron.fasterRcnnCamera import fasterRcnnRealTimeDetect


def openNewTab(windowTitle,imgPath):
    # Create a new window
    new_window = tk.Toplevel()
    new_window.title(windowTitle)

    # Load the image (replace 'path/to/your/image.jpg' with the actual path)
    img = Image.open(imgPath)
    img = img.resize((640, 640), Image.ANTIALIAS)  # Resize the image if needed
    photo = ImageTk.PhotoImage(img)

    # Create a label to display the image
    image_label = tk.Label(new_window, image=photo)
    image_label.image = photo  # Keep a reference to prevent garbage collection
    image_label.pack()


def yoloPhotoDetection():
    fileTypes = [("Image files", "*.png;*.jpg;*.jpeg")]
    path = tk.filedialog.askopenfilename(filetypes=fileTypes)

    # if file is selected
    if len(path):
        detectionResult = yoloDetectPhoto(path)
        img = Image.open(detectionResult[1])
        img = img.resize((460, 460), Image.LANCZOS)  # Resize the image if needed
        photo = ImageTk.PhotoImage(img)

        # Create a label to display the image
        image_label = tk.Label(app, image=photo)
        image_label.image = photo  # Keep a reference to prevent garbage collection
        image_label.place(relx=0.5, rely=0.55, anchor=tk.CENTER)  # Position the image label

        # Create a label to display the detection text
        detection_text = tk.Label(app, text=detectionResult[0], font=("Helvetica", 16))
        detection_text.place(relx=0.5, rely=0.88, anchor=tk.CENTER)  # Position the detection text label

    # if no file is selected, then we are displaying below message
    else:
        print("No file is chosen !! Please choose a file.")

def ssdPhotoDetection():
    fileTypes = [("Image files", "*.png;*.jpg;*.jpeg")]
    path = tk.filedialog.askopenfilename(filetypes=fileTypes)

    # if file is selected
    if len(path):
        detectionResult = ssdDetectPhoto(path)
        img = Image.open(detectionResult[1])
        img = img.resize((460, 460), Image.LANCZOS)  # Resize the image if needed
        photo = ImageTk.PhotoImage(img)

        # Create a label to display the image
        image_label = tk.Label(app, image=photo)
        image_label.image = photo  # Keep a reference to prevent garbage collection
        image_label.place(relx=0.5, rely=0.55, anchor=tk.CENTER)  # Position the image label

        # Create a label to display the detection text
        detection_text = tk.Label(app, text=detectionResult[0], font=("Helvetica", 16))
        detection_text.place(relx=0.5, rely=0.88, anchor=tk.CENTER)  # Position the detection text label


    # if no file is selected, then we are displaying below message
    else:
        print("No file is chosen !! Please choose a file.")


def fasterRcnnPhotoDetection():
    fileTypes = [("Image files", "*.png;*.jpg;*.jpeg")]
    path = tk.filedialog.askopenfilename(filetypes=fileTypes)

    # if file is selected
    if len(path):
        detectionResult = fasterRcnnDetectPhoto(path)
        img = Image.open(detectionResult[1])
        img = img.resize((460, 460), Image.LANCZOS)  # Resize the image if needed
        photo = ImageTk.PhotoImage(img)

        # Create a label to display the image
        image_label = tk.Label(app, image=photo)
        image_label.image = photo  # Keep a reference to prevent garbage collection
        image_label.place(relx=0.5, rely=0.55, anchor=tk.CENTER)  # Position the image label

        # Create a label to display the detection text
        detection_text = tk.Label(app, text=detectionResult[0], font=("Helvetica", 16))
        detection_text.place(relx=0.5, rely=0.88, anchor=tk.CENTER)  # Position the detection text label

    # if no file is selected, then we are displaying below message
    else:
        print("No file is chosen !! Please choose a file.")

def yoloRealTimeDetection():
    yoloRealTimeDetect()

def ssdRealTimeDetection():
    ssdRealTimeDetect()

def fasterRcnnRealTimeDetection():
    fasterRcnnRealTimeDetect()

if __name__ == "__main__":
    # defining tkinter object
    app = tk.Tk()
    app.title("Car Plate Number Recognition System")
    app.geometry("1000x800")

    # Adding background color to labels and buttons
    app.option_add("*Label*Background", "white")
    app.option_add("*Button*Background", "lightgreen")

    # Create a notebook (tabbed interface)
    notebook = ttk.Notebook(app)

    # Create frames for each model
    yolo_frame = ttk.Frame(notebook)
    ssd_frame = ttk.Frame(notebook)
    fasterRcnn_frame = ttk.Frame(notebook)

    # Add frames to the notebook
    notebook.add(yolo_frame, text="YOLO")
    notebook.add(ssd_frame, text="SSD")
    notebook.add(fasterRcnn_frame, text="Faster Rcnn")

    # Customize tab appearance (increase font size and padding)
    style = ttk.Style()
    style.configure("TNotebook.Tab", font=("Helvetica", 14), padding=[20, 10])

    # Create buttons for each model within their respective frames
    yolo_button1 = tk.Button(yolo_frame, text="Detect Image (YOLO)", command=yoloPhotoDetection, font=("Helvetica", 16),
                             width=25, height=2)  # Adjust width, height, and font size as needed
    yolo_button2 = tk.Button(yolo_frame, text="Live Detect (YOLO)", command=yoloRealTimeDetection,
                             font=("Helvetica", 16), width=25,
                             height=2)  # Adjust width, height, and font size as needed

    ssd_button1 = tk.Button(ssd_frame, text="Detect Image (SSD)", command=ssdPhotoDetection, font=("Helvetica", 16),
                            width=25, height=2)
    ssd_button2 = tk.Button(ssd_frame, text="Live Detect (SSD)", command=ssdRealTimeDetection, font=("Helvetica", 16),
                            width=25, height=2)

    fasterRcnn_button1 = tk.Button(fasterRcnn_frame, text="Detect Image (Detectron2)", command=fasterRcnnPhotoDetection,
                                   font=("Helvetica", 16), width=25, height=2)
    fasterRcnn_button2 = tk.Button(fasterRcnn_frame, text="Live Detect (Detectron2)",
                                   command=fasterRcnnRealTimeDetection, font=("Helvetica", 16), width=25, height=2)

    # Create a label to display the image
    image_label = tk.Label(app)
    image_label.place(x=20, y=20)  # Position the image label

    # Create a label to display the detection text
    detection_text = tk.Label(app, font=("Helvetica", 16))
    detection_text.place(x=20, y=660)  # Position the detection text label

    # Pack the buttons with customized positions
    yolo_button1.place(x=150, y=50)
    yolo_button2.place(x=500, y=50)

    ssd_button1.place(x=150, y=50)
    ssd_button2.place(x=500, y=50)

    fasterRcnn_button1.place(x=150, y=50)
    fasterRcnn_button2.place(x=500, y=50)

    # Pack the notebook
    notebook.pack(fill="both", expand=True)

    app.mainloop()

