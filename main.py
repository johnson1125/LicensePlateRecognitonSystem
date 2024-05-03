import tkinter as tk
from tkinter import ttk
from tkinter import Label
from tkinter import filedialog
from PIL import Image, ImageTk
from models.yolo.yoloPhoto import yoloDetectPhoto
from models.yolo.yoloRealTimeDetection import yoloRealTimeModelDetect
from models.sdd_MobileNetV2_FpnLite.sddMobileNetV2Photo import ssdDetectPhoto
from models.sdd_MobileNetV2_FpnLite.ssdMobileNetV2RealTimeDetection import ssdRealTimeModelDetect
from models.faster_rcnn.fasterRcnnPhoto import fasterRcnnDetectPhoto
from models.faster_rcnn.fasterRcnnCamera import fasterRcnnRealTimeDetect
from datetime import datetime

def yoloPhotoDetection():
    fileTypes = [("Image files", "*.png;*.jpg;*.jpeg")]
    path = tk.filedialog.askopenfilename(filetypes=fileTypes)

    # if file is selected
    if len(path):
        detectionResult = yoloDetectPhoto(path)
        if detectionResult is None:
            image_label_yolo.config(image=empty_photo)
            detection_text_yolo.config(text= "No car plate detected...", foreground="red" )
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
            image_label_ssd.config(image=empty_photo)
            detection_text_ssd.config(text= "No car plate detected...", foreground="red" )
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
            image_label_faster_rcnn.config(image=empty_photo)
            detection_text_faster_rcnn.config(text= "No car plate detected...", foreground="red" )
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
    yoloResult = yoloRealTimeModelDetect()

    if(yoloResult is not None):
        detectionLive_text_yolo.config(text=f"Plate Number: {yoloResult} : Entered at {datetime.now()}")
        # Append user input to a text file
        with open("resources/entered_record.txt", "a") as file:
            file.write(f"Plate Number: {yoloResult} : Entered at {datetime.now()}" + "\n")

def ssdRealTimeDetection():
    ssdResult = ssdRealTimeModelDetect()

    if (ssdResult is not None):
        detectionLive_text_ssd.config(text=f"Plate Number: {ssdResult} : Entered at {datetime.now()}")
        with open("resources/entered_record.txt", "a") as file:
            file.write(f"Plate Number: {ssdResult} : Entered at {datetime.now()}" + "\n")
def fasterRcnnRealTimeDetection():
    fasterRcnnResult = fasterRcnnRealTimeDetect()

    if (fasterRcnnResult is not None):
        detectionLive_text_faster_rcnn.config(text=f"Plate Number: {fasterRcnnResult} : Entered at {datetime.now()}")
        with open("resources/entered_record.txt", "a") as file:
            file.write(f"Plate Number: {fasterRcnnResult} : Entered at {datetime.now()}" + "\n")

def on_button_click():
    user_input = registerEntry.get().upper()
    print("User Input:", user_input)

    # Append user input to a text file
    with open("resources/registered_car_plate.txt", "a") as file:
        file.write(user_input + "\n")
        print("User input appended to file.")

    # Clear the text in the entry widget
    registerEntry.delete(0, 'end')

def clear_displayed_content():
    # Clear image labels
    image_label_yolo.config(image=empty_photo)
    image_label_ssd.config(image=empty_photo)
    image_label_faster_rcnn.config(image=empty_photo)

    # Clear detection text labels
    detection_text_yolo.config(text="")
    detection_text_ssd.config(text="")
    detection_text_faster_rcnn.config(text="")

    detectionLive_text_yolo.config(text="")
    detectionLive_text_ssd.config(text="")
    detectionLive_text_faster_rcnn.config(text="")

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

    # Bind the function to the tab changing event
    notebook.bind("<<NotebookTabChanged>>", lambda event: clear_displayed_content())

    # Customize tab appearance (increase font size and padding)
    style = ttk.Style()
    style.configure("TNotebook.Tab", font=("Helvetica", 14), padding=[20, 10])

    # Create a label text provide instruction to user
    instruction_text_yolo = tk.Label(yoloLive_frame, text="Click on the button to open Camera", font=("Helvetica", 16))
    instruction_text_yolo.place(relx=0.5, rely=0.07, anchor=tk.CENTER)

    instruction_text_yolo1 = tk.Label(yolo_frame, text="Click on the button to upload image", font=("Helvetica", 16))
    instruction_text_yolo1.place(relx=0.5, rely=0.07, anchor=tk.CENTER)

    instruction_text_ssd = tk.Label(ssdLive_frame, text="Click on the button to open Camera", font=("Helvetica", 16))
    instruction_text_ssd.place(relx=0.5, rely=0.07, anchor=tk.CENTER)

    instruction_text_ssd1 = tk.Label(ssd_frame, text="Click on the button to upload image", font=("Helvetica", 16))
    instruction_text_ssd1.place(relx=0.5, rely=0.07, anchor=tk.CENTER)

    instruction_text_faster = tk.Label(fasterRcnnLive_frame, text="Click on the button to open Camera",
                                       font=("Helvetica", 16))
    instruction_text_faster.place(relx=0.5, rely=0.07, anchor=tk.CENTER)

    instruction_text_faster1 = tk.Label(fasterRcnn_frame, text="Click on the button to upload image",
                                       font=("Helvetica", 16))
    instruction_text_faster1.place(relx=0.5, rely=0.07, anchor=tk.CENTER)

    # Create buttons for each model within their respective frames
    yolo_button1 = tk.Button(yolo_frame, text="Detect Image (YOLO)",
                             command=yoloPhotoDetection, font=("Helvetica", 16), width=25, height=2)
    yolo_button1.place(relx=0.5, rely=0.15, anchor=tk.CENTER)

    ssd_button1 = tk.Button(ssd_frame, text="Detect Image (SSD)",
                            command=ssdPhotoDetection, font=("Helvetica", 16), width=25, height=2)
    ssd_button1.place(relx=0.5, rely=0.15, anchor=tk.CENTER)

    fasterRcnn_button1 = tk.Button(fasterRcnn_frame, text="Detect Image (FasterRCNN)",
                                   command=fasterRcnnPhotoDetection, font=("Helvetica", 16), width=25, height=2)
    fasterRcnn_button1.place(relx=0.5, rely=0.15, anchor=tk.CENTER)

    yolo_button2 = tk.Button(yoloLive_frame, text="Live Detect (YOLO)",
                             command=yoloRealTimeDetection,font=("Helvetica", 16), width=25, height=2)
    yolo_button2.place(relx=0.5, rely=0.15, anchor=tk.CENTER)

    ssd_button2 = tk.Button(ssdLive_frame, text="Live Detect (SSD)",
                            command=ssdRealTimeDetection, font=("Helvetica", 16), width=25, height=2)
    ssd_button2.place(relx=0.5, rely=0.15, anchor=tk.CENTER)

    fasterRcnn_button2 = tk.Button(fasterRcnnLive_frame, text="Live Detect (FasterRCNN)",
                                   command=fasterRcnnRealTimeDetection, font=("Helvetica", 16), width=25, height=2)
    fasterRcnn_button2.place(relx=0.5, rely=0.15, anchor=tk.CENTER)

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

    # Create a label to display the detection text for each frame
    detectionLive_text_yolo = tk.Label(yoloLive_frame, font=("Helvetica", 16))
    detectionLive_text_yolo.place(relx=0.5, rely=0.3, anchor=tk.CENTER)

    detectionLive_text_ssd = tk.Label(ssdLive_frame, font=("Helvetica", 16))
    detectionLive_text_ssd.place(relx=0.5, rely=0.3, anchor=tk.CENTER)

    detectionLive_text_faster_rcnn = tk.Label(fasterRcnnLive_frame, font=("Helvetica", 16))
    detectionLive_text_faster_rcnn.place(relx=0.5, rely=0.3, anchor=tk.CENTER)

    # Create a label for the text input (register)
    registerLabel1 = tk.Label(register_frame, text="Register Car Plate Number Here", font=("Helvetica", 30))
    registerLabel1.place(relx=0.5, rely=0.1, anchor=tk.CENTER)

    # Create a label for the text input (register)
    registerLabel2 = tk.Label(register_frame, text="Enter your car plate number:", font=("Helvetica", 16))
    registerLabel2.place(relx=0.5, rely=0.2, anchor=tk.CENTER)

    # Create a text input field (register)
    registerEntry = tk.Entry(register_frame,  font=("Helvetica", 16))
    registerEntry.place(relx=0.5, rely=0.25, anchor=tk.CENTER)

    # Create a button (register)
    registerButton = tk.Button(register_frame, text="Submit", command=on_button_click,  font=("Helvetica", 16))
    registerButton.place(relx=0.5, rely=0.30, anchor=tk.CENTER)

    # Create an empty photo object
    empty_photo = ImageTk.PhotoImage(Image.new('RGB', (1, 1)))

    # Pack the notebook
    notebook.pack(fill="both", expand=True)

    app.mainloop()

