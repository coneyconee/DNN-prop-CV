#libraries
import os               #for checking if the screenshots folder exists and to make the folder if it does not exist
import cv2              #for facebox DNN
import dlib             #for face landmarks DNN
import mediapipe as mp  #for hand landmarks CNN
import datetime         #photo filename
import time             #photo buffer
import numpy as np      #type conversion
import imutils          #for rotation of image
import tkinter as tk    #for ui
from tkmacosx import Button #for buttons because tkinter buttons don't work on macos
from PIL import Image, ImageTk #for converting cv2 images into a format that can be processed by tkinter

modelPath = "res10_300x300_ssd_iter_140000.caffemodel"                                              #model for DNN face detection
configPath = "deploy.prototxt"                                                                      #config ^^
net = cv2.dnn.readNetFromCaffe(configPath, modelPath)                                           
landmarkPredictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")                   #dlib model for DNN face landmarks prediction
mpHands = mp.solutions.hands                                                                        #mediapipe CNN for hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=10, min_detection_confidence=0.2)      #arguments ^^

#conts for snapshot capture capability
lastCaptureTime = 0
captureCooldown = 2

#glasses is headprop, ring is handprop
headprops = [cv2.imread("assets/head{}.png".format(i), cv2.IMREAD_UNCHANGED) for i in range(1,4)] # loads head1, head2 and head3
handprops = [cv2.resize(cv2.imread("assets/hand{}.png".format(i), cv2.IMREAD_UNCHANGED),(300,300)) for i in range(1,4)] # loads head1, head2 and head3
glasses = headprops[0]
ring = handprops[0]
#force size to 300^2 to maintain overlay size
# ring = cv2.resize(cv2.imread("assets/hand1.png", cv2.IMREAD_UNCHANGED),(300,300)) #force size to 300^2 to maintain overlay size
ring = imutils.rotate(ring, 0) #in case u want to rotate prop

#video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)

#overlays an image with an alpha channel onto the stream, and also applying perspective
def overlayTransparentPerspective(bg, overlayImg, pts1, pts2):
    #case 1: no prop image provided
    if overlayImg is None or overlayImg.shape[0] == 0 or overlayImg.shape[1] == 0:
        print("No prop image")
        return bg
    
    #using origin points (pt1) and face points (pt2) and cv2.warpPerspective to stretch the image to the face
    M = cv2.getPerspectiveTransform(np.float32(pts1), np.float32(pts2))
    h, w = bg.shape[:2]
    overlayWarped = cv2.warpPerspective(overlayImg, M, (w, h))

    mask = overlayWarped[:, :, 3:] / 255.0
    overlayImage = overlayWarped[:, :, :3]

    #case 2: prop exceeds window
    roi = bg[:overlayWarped.shape[0], :overlayWarped.shape[1]]
    if roi.shape[:2] != mask.shape[:2]:
        print("Size mismatch")
        return bg

    #case 3: for camera flash fx, have an additional channel
    bg[:overlayWarped.shape[0], :overlayWarped.shape[1]] = (1 - mask) * roi + mask * overlayImage
    return bg

#check if the palm is closed by checking if the tips of all fingers are below the knuckle of all fingers
def palmClose(handLandmarks):
    fingers = [
        (mpHands.HandLandmark.INDEX_FINGER_TIP, mpHands.HandLandmark.INDEX_FINGER_MCP),
        (mpHands.HandLandmark.MIDDLE_FINGER_TIP, mpHands.HandLandmark.MIDDLE_FINGER_MCP),
        (mpHands.HandLandmark.RING_FINGER_TIP, mpHands.HandLandmark.RING_FINGER_MCP),
        (mpHands.HandLandmark.PINKY_TIP, mpHands.HandLandmark.PINKY_MCP)
    ]
    return all(handLandmarks.landmark[tip].y >= handLandmarks.landmark[mcp].y for tip, mcp in fingers)

#sart video capture stream loop
# def dnn_loop(headprop, handprop):
def dnn_loop(headprop,handprop):
    # sets the prop
    global glasses, ring, headprops, handprops
    glasses = headprops[headprop-1]
    ring = handprops[handprop-1]

    ret, frame = cap.read()
    if not ret:
        print('Error: video feed cannot be found, crashing...')
        cap.release()
        exit()

    #flip frame to mirror like an iphone
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    #face detection DNN (not shown at all on screen)
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    #get the confidence of the face detection
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        #threshold of confidence for the dlib landmarks DNN to be activated
        if confidence > 0.4:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (startX, startY, endX, endY) = box.astype("int")

            faceRoi = dlib.rectangle(startX, startY, endX, endY)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            landmarks = landmarkPredictor(gray, faceRoi)

        #threshold of confidence for the faceprop to be overlayed onto the dlib landmarks
        if confidence > 0.60:
            for n in range(68):
                xVal, yVal = landmarks.part(n).x, landmarks.part(n).y

            #pts1 is the origin points of the overlay image (prop) which would be the 4 corners of the image (if its transparent, the alpha channel has to be padded)
            pts1 = np.array([
                [0, 0],
                [glasses.shape[1], 0],
                [0, glasses.shape[1]],
                [glasses.shape[1], glasses.shape[1]]
            ], dtype=np.float32)

            #define the eyes from landmarks
            eyeLeft = np.array([landmarks.part(36).x, landmarks.part(36).y])
            eyeRight = np.array([landmarks.part(45).x, landmarks.part(45).y])

            #find the angle between the eyes (impt for next step)
            dx = eyeRight[0] - eyeLeft[0]
            dy = eyeRight[1] - eyeLeft[1]
            angle = np.arctan2(dy, dx)

            #since we are using only 3 points on the face, eyes and chin, for the chin we interpolate 2 points on either side
            #think of it as having rods pivoting from the chin horizontally, and rods pivoting from the eyes vertically that connect at a point
            #the point of connection will be our 2 bottom points that follow the x-axis of the eyes, and the y-axis of the eyes with displacement, while being constrained on the y-axis by the centre chin point
            chinCenter = np.array([landmarks.part(8).x, landmarks.part(8).y])

            chinLeft = np.array([
                chinCenter[0] - (eyeRight[0] - eyeLeft[0]) // 2,
                chinCenter[1]
            ])
            chinRight = np.array([
                chinCenter[0] + (eyeRight[0] - eyeLeft[0]) // 2,
                chinCenter[1]
            ])

            def rotatePoint(point, center, angleRad):
                xCoord, yCoord = point
                cx, cy = center
                cosA, sinA = np.cos(angleRad), np.sin(angleRad)
                xNew = cx + cosA * (xCoord - cx) - sinA * (yCoord - cy)
                yNew = cy + sinA * (xCoord - cx) + cosA * (yCoord - cy)
                return np.array([xNew, yNew])

            chinLeftRotated = rotatePoint(chinLeft, chinCenter, angle)
            chinRightRotated = rotatePoint(chinRight, chinCenter, angle)

            pts2 = np.array([eyeLeft, eyeRight, chinLeftRotated, chinRightRotated], dtype=np.float32)

            glassesResized = cv2.resize(glasses, (glasses.shape[1], glasses.shape[1]))

            #this offset is just in case you want to have a hat prop that is displaced away from the centre of the face (negative = up)
            hatoffset = -200

            pts2[:, 1] += hatoffset

            #overlay the faceprop
            frame = overlayTransparentPerspective(frame, glassesResized, pts1, pts2)

    #preprocess video stream for the hand landmarks
    rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgbFrame)

    #if hands are present
    if result.multi_hand_landmarks:
        for handLandmarks in result.multi_hand_landmarks:
            #extract key landmarks for mapping prop
            #wrist helps to
            wrist = handLandmarks.landmark[mpHands.HandLandmark.WRIST]
            middleMcp = handLandmarks.landmark[mpHands.HandLandmark.MIDDLE_FINGER_MCP]
            indexMcp = handLandmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_MCP]
            pinkyMcp = handLandmarks.landmark[mpHands.HandLandmark.PINKY_MCP]
            indexPip = handLandmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_PIP]
            pinkyPip = handLandmarks.landmark[mpHands.HandLandmark.PINKY_PIP]
            #find pixels for these landmarks
            wristX, wristY = int(wrist.x * w), int(wrist.y * h)
            middleX, middleY = int(middleMcp.x * w), int(middleMcp.y * h)
            indexX, indexY = int(indexMcp.x * w), int(indexMcp.y * h)
            pinkyX, pinkyY = int(pinkyMcp.x * w), int(pinkyMcp.y * h)
            indexPipX, indexPipY = int(indexPip.x * w), int(indexPip.y * h)
            pinkyPipX, pinkyPipY = int(pinkyPip.x * w), int(pinkyPip.y * h)

            #4 corners of the prop (origin img)
            pts1 = np.array([
                [0, 0],
                [ring.shape[1], 0],
                [0, ring.shape[0]],
                [ring.shape[1], ring.shape[0]]
            ], dtype=np.float32)
 
            #find hand width by measuring distance between index and pinky
            widthTop = np.linalg.norm(np.array([indexX, indexY]) - np.array([pinkyX, pinkyY]))
            widthBottom = np.linalg.norm(np.array([indexPipX, indexPipY]) - np.array([pinkyPipX, pinkyPipY]))
            avgWidth = (widthTop + widthBottom) / 2

            #find hand height by measuring distances between index and pinky finger tip joints respectively and finding the avg
            heightLeft = np.linalg.norm(np.array([indexX, indexY]) - np.array([indexPipX, indexPipY]))
            heightRight = np.linalg.norm(np.array([pinkyX, pinkyY]) - np.array([pinkyPipX, pinkyPipY]))
            avgHeight = (heightLeft + heightRight) / 2

            #index and pinky and their subsidiary bottom points which will be the 4 points the overlay prop will be mapped directly to
            #they have to be subsidiary because we need almost a perfect square for the image to not look stretched weirdly
            pts2 = np.array([
                [indexX, indexY],
                [pinkyX, pinkyY],
                [indexX, indexY + int(avgHeight)],
                [pinkyX, pinkyY + int(avgHeight)]
            ], dtype=np.float32)

            #aspect ratio of the prop so we can keep it in scale
            propAspect = ring.shape[1] / ring.shape[0]

            #trivial
            #hand W and H to scale the prop to
            handWidth = np.linalg.norm(np.array([indexX, indexY]) - np.array([pinkyX, pinkyY]))
            handHeight = np.linalg.norm(np.array([wristX, wristY]) - np.array([middleX, middleY]))
            scaleFactor = handWidth / ring.shape[1]
            if handHeight / handWidth > propAspect:
                scaleFactor = handHeight / ring.shape[0]

            #multiplier for the size (arbitrary 2 value but it stays untouched mostly)
            sizeMult = 2
            scaledWidth = int(ring.shape[1] * scaleFactor * sizeMult)
            scaledHeight = int(ring.shape[0] * scaleFactor * sizeMult)

            #map the overlay image to the correct dimensions
            ringResized = cv2.resize(ring, (scaledWidth, scaledHeight))

            #use the above ^^ dimensions and cv2.warpPerspective() the prop corners to the hand points
            frame = overlayTransparentPerspective(frame, ringResized, pts1, pts2)

            #palm close
            #use time for buffer
            currTime = time.time()
            global lastCaptureTime # to sync variable from both inside and outside the function
            #if palm is closed and the cooldown is over
            if palmClose(handLandmarks) and (currTime - lastCaptureTime > captureCooldown):
                #set new capture time for buffer
                lastCaptureTime = currTime
                #add a timestamp of the date to the filename using datetime (chatGPTed this)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                #add suffix
                filename = f"capture_{timestamp}.png"

                #check if the folder exists, and if not, creates it
                if not os.path.isdir('screenshots'):
                    os.mkdir('screenshots')
                #make a capture
                cv2.imwrite(('screenshots/'+filename), frame)
                print(f"SS saved as {filename} in folder: screenshots")

                #use numpy to overlay a white screen momentarily to simulate a camera flash
                flash = np.ones_like(frame, dtype=np.uint8) * 255
                alphaVal = 0.32
                flashFrame = cv2.addWeighted(frame, 1 - alphaVal, flash, alphaVal, 0)
                frame = flashFrame
    return ret, frame

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

headprop = 1
handprop = 1

def update_frame():
    # ret, frame = cap.read()
    global headprop, handprop
    ret, frame = dnn_loop(headprop,handprop)
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


# import props and convert to imagetk format for tkinter to process
head1 = ImageTk.PhotoImage((Image.open("assets/head1.png")).resize((110, 110)))
head2 = ImageTk.PhotoImage((Image.open("assets/head2.png")).resize((110, 110)))
head3 = ImageTk.PhotoImage((Image.open("assets/head3.png")).resize((110, 110)))
hand1 = ImageTk.PhotoImage((Image.open("assets/hand1.png")).resize((110, 110)))
hand2 = ImageTk.PhotoImage((Image.open("assets/hand2.png")).resize((110, 110)))
hand3 = ImageTk.PhotoImage((Image.open("assets/hand3.png")).resize((110, 110)))

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

#labelling which buttons are for head props and which are for hand props
headlabel = tk.Label(window, 
                 text='HEAD PROPS', 
                 anchor=tk.CENTER,       
                 bg = '#DAE4FF',
                 fg = 'black',
                 height=3,              
                 width=13,                  
                 font=("Arial", 16, "bold"))

handlabel = tk.Label(window, 
                 text='HAND PROPS', 
                 anchor=tk.CENTER,       
                 bg = '#DAE4FF',
                 fg = 'black',
                 height=3,              
                 width=13,                  
                 font=("Arial", 16, "bold"))

headlabel.place(x=0,y=0)
handlabel.place(x=1155,y=0)
window.mainloop()

cap.release()