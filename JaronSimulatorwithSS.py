#libs
import cv2
import dlib
import mediapipe as mp
import datetime
import time
import numpy as np
import math

#config models
model_path = "res10_300x300_ssd_iter_140000.caffemodel"
config_path = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(config_path, model_path)
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=10, min_detection_confidence=0.2)

#constants for capture
last_capture_time = 0
capture_cooldown = 2

#prop src's
glasses = cv2.imread("jaron.jpg", cv2.IMREAD_UNCHANGED)  
ring = cv2.imread("800d.png", cv2.IMREAD_UNCHANGED)        

#config video stream
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)

#overlaying props onto landmarks
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
 
#detect palm close for capture
def palmclose(hand_landmarks):

    fingers = [
        (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_MCP),
        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP),
        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_MCP),
        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_MCP)
    ]

    closed = True
    for tip, mcp in fingers:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[mcp].y:
            closed = False

    
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    thumb_near_palm = abs(thumb_tip.x - thumb_mcp.x) < 0.1  

    return closed and thumb_near_palm  

#begin video stream in window
while True:
    ret, frame = cap.read()
    if not ret:
        break

    #flip
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    #DNN facebox detection using OpenCV
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        #threshold for facebox
        if confidence > 0.4:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (startX, startY, endX, endY) = box.astype("int")
            
            #DNN face landmarks using D Library
            face_roi = dlib.rectangle(startX, startY, endX, endY)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            landmarks = landmark_predictor(gray, face_roi)

        #threshold for face landmarks
        if confidence > 0.65:
            
            for n in range(68):
                x, y = landmarks.part(n).x, landmarks.part(n).y

            
            left_eye_x = landmarks.part(36).x  
            left_eye_y = landmarks.part(36).y
            right_eye_x = landmarks.part(45).x  
            right_eye_y = landmarks.part(45).y

            glasses_width = right_eye_x - left_eye_x + 200
            glasses_height = int(glasses_width) 
            glasses_x = left_eye_x - int(glasses_width * 0.2)
            glasses_y = left_eye_y - int(glasses_height * 0.5)
        

            glasses_x = max(0, min(glasses_x, w - glasses_width))
            glasses_y = max(0, min(glasses_y, h - glasses_height))

            frame = overlay_transparent(frame, glasses, glasses_x, glasses_y, (glasses_width, glasses_height))


    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            index_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x * w)
            index_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * h)

            wrist_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * w)
            wrist_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * h)

            middle_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * w)
            middle_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * h)

            thumb_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
            pinky_x = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x

            is_left_hand = thumb_x > pinky_x

            if is_left_hand:
                index_x -= 200
                index_y -= 100
            else:
                index_x += 15
                index_y -= 100
                
            ring_size = (int(330 * ((int(wrist_y) - int(middle_y))/200)), int(330 * ((int(wrist_y) - int(middle_y))/200)))

            index_x = max(0, min(index_x, w - ring_size[0]))
            index_y = max(0, min(index_y, h - ring_size[1]))

            frame = overlay_transparent(frame, ring, index_x, index_y, ring_size)

            current_time = time.time()
            if palmclose(hand_landmarks) and (current_time - last_capture_time > capture_cooldown):
                last_capture_time = current_time
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"capture_{timestamp}.png"
                cv2.imwrite(filename, frame)
                print(f"SS saved as {filename}")

                flash = np.ones_like(frame, dtype=np.uint8) * 255
                alpha = 0.75
                flash_frame = cv2.addWeighted(frame, 1 - alpha, flash, alpha, 0)
                cv2.imshow("JARON SIMULATOR", flash_frame)
                cv2.waitKey(200)



    cv2.imshow("JARON SIMULATOR", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()