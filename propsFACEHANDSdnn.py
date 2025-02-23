
import cv2
import dlib
import time
import mediapipe as mp
import tkinter as tk
from tkmacosx import Button
import cv2
from PIL import Image, ImageTk
import numpy as np

model_path = "res10_300x300_ssd_iter_140000.caffemodel"
config_path = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(config_path, model_path)

landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=10, min_detection_confidence=0.2)

glasses = cv2.imread("jaron.jpg", cv2.IMREAD_UNCHANGED)  
# ring = cv2.imread("800d.png", cv2.IMREAD_UNCHANGED)      

# defining props used
headprop = 1
handprop = 1

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)


def overlay_transparent(background, overlay, x, y, overlay_size=None):
    if overlay is None or overlay.shape[0] == 0 or overlay.shape[1] == 0:
        print("no prop image")
        return background  

    if overlay_size is not None:
        overlay = cv2.resize(overlay, overlay_size, interpolation=cv2.INTER_AREA)

    h, w = overlay.shape[:2]
    if x + w > background.shape[1] or y + h > background.shape[0]:
        return background  

    
    if overlay.shape[2] == 3:
        print("no alpha channel")
        b, g, r = cv2.split(overlay)
        alpha = np.ones(b.shape, dtype=b.dtype) * 255  
        overlay = cv2.merge((b, g, r, alpha))

    
    overlay_image = overlay[:, :, :3]  
    mask = overlay[:, :, 3:] / 255.0   

    
    roi = background[y:y+h, x:x+w]

    
    if roi.shape[:2] != mask.shape[:2]:
        print("size mismatch")
        return background

    
    background[y:y+h, x:x+w] = (1 - mask) * roi + mask * overlay_image

    return background
 



def dnnLoop(headprop, handprop):
    ret, frame = cap.read()
    if not ret:
        cap.release()
        exit()

    
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        
        if confidence > 0.4:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (startX, startY, endX, endY) = box.astype("int")

            
            text = f"Confidence: {confidence:.2f}"
            '''cv2.putText(frame, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)'''
            face_roi = dlib.rectangle(startX, startY, endX, endY)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            landmarks = landmark_predictor(gray, face_roi)

        
        if confidence > 0.75:
            
            for n in range(68):
                x, y = landmarks.part(n).x, landmarks.part(n).y
                #cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            
            left_eye_x = landmarks.part(36).x  
            left_eye_y = landmarks.part(36).y
            right_eye_x = landmarks.part(45).x  
            right_eye_y = landmarks.part(45).y

            glasses_width = right_eye_x - left_eye_x + 200
            glasses_height = int(glasses_width) 
            glasses_x = left_eye_x - int(glasses_width * 0.2)
            glasses_y = left_eye_y - int(glasses_height * 0.5)

            frame = overlay_transparent(frame, glasses, glasses_x, glasses_y, (glasses_width, glasses_height))

    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
               # cv2.circle(frame, (cx, cy), 5, (255, 255, 0), -1)

            
 #           mp.solutions.drawing_utils.draw_landmarks(
  #              frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
   #             mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
    #            mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2)
     #       )

            
            index_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x * w)
            index_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * h)

            ring_size = (250, 250)  
            # frame = overlay_transparent(frame, ring, index_x - 15, index_y - 15, ring_size)

    return ret, frame

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     exit()


# ------------------------UI-------------------------------

window = tk.Tk()
window.title("SST Photobooth")
window.geometry("1280x720")
window.minsize(1280, 720)
window.maxsize(1280, 720)
window.config(background = "#DAE4FF")

# Create a label to display the video feed
label = tk.Label(window,width=1020)
label.place(x=130)

def update_frame():
    # ret, frame = cap.read()
    global headprop, handprop
    ret, frame = dnnLoop(headprop,handprop)
    if ret:
        # Convert frame from BGR to RGB becasue cv2 outputs in BGR and PIL takes RGB input
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert frame to PIL Image for PIL to process
        img = Image.fromarray(frame)
        # Convert Image to ImageTk format for tkinter to process
        imgtk = ImageTk.PhotoImage(image=img)
        # Update label with new image
        label.imgtk = imgtk  # Keep reference to avoid garbage collection
        label.configure(image=imgtk)
    # Call this function again after 10ms
    window.after(10, update_frame)

# Start the video update loop
update_frame()

def updatehandprop(part,propnum):
    global headprop, handprop
    if part == 'head':
        headprop = propnum
    elif part == 'hand':
        handprop = propnum
    print(headprop,handprop)


# import props and convert to imagetk format for tkinter to process
head1 = ImageTk.PhotoImage((Image.open("assets/head1.png")).resize((160, 90)))
head2 = ImageTk.PhotoImage((Image.open("assets/head2.png")).resize((160, 90)))
head3 = ImageTk.PhotoImage((Image.open("assets/head3.png")).resize((160, 90)))
hand1 = ImageTk.PhotoImage((Image.open("assets/hand1.png")).resize((160, 90)))
hand2 = ImageTk.PhotoImage((Image.open("assets/hand2.png")).resize((160, 90)))
hand3 = ImageTk.PhotoImage((Image.open("assets/hand3.png")).resize((160, 90)))

# prop buttons
headprop1 = Button(window,
                width=130,
                fg = "#DAE4FE", bg = "#DAE4FF",
                borderless = 1,
                activebackground = ("#AFB7CD"),
                highlightbackground = "#DAE4FE",
                focusthickness = 0,
                image=head1,
                command = lambda: updatehandprop('head',1)
)
headprop2 = Button(window,
                width=130,
                fg = "#DAE4FE", bg = "#DAE4FF",
                borderless = 1,
                activebackground = ("#AFB7CD"),
                highlightbackground = "#DAE4FE",
                focusthickness = 0,
                image=head2,
                command = lambda: updatehandprop('head',2)
)
headprop3 = Button(window,
                width=130,
                fg = "#DAE4FE", bg = "#DAE4FF",
                borderless = 1,
                activebackground = ("#AFB7CD"),
                highlightbackground = "#DAE4FE",
                focusthickness = 0,
                image=head3,
                command = lambda: updatehandprop('head',3)
)

handprop1 = Button(window,
                width=130,
                fg = "#DAE4FE", bg = "#DAE4FF",
                borderless = 1,
                activebackground = ("#AFB7CD"),
                highlightbackground = "#DAE4FE",
                focusthickness = 0,
                image=hand1,
                command = lambda: updatehandprop('hand',1)
)
handprop2 = Button(window,
                width=130,
                fg = "#DAE4FE", bg = "#DAE4FF",
                borderless = 1,
                activebackground = ("#AFB7CD"),
                highlightbackground = "#DAE4FE",
                focusthickness = 0,
                image=hand2,
                command = lambda: updatehandprop('hand',2)
)
handprop3 = Button(window,
                width=130,
                fg = "#DAE4FE", bg = "#DAE4FF",
                borderless = 1,
                activebackground = ("#AFB7CD"),
                highlightbackground = "#DAE4FE",
                focusthickness = 0,
                image=hand3,
                command = lambda: updatehandprop('hand',3)
)

headprop1.place(x=0,y=105)
headprop2.place(x=0,y=315)
headprop3.place(x=0,y=525)
handprop1.place(x=1155,y=105)
handprop2.place(x=1155,y=315)
handprop3.place(x=1155,y=525)

window.mainloop()

cap.release()