
import cv2
import dlib
import time
import mediapipe as mp

model_path = "res10_300x300_ssd_iter_140000.caffemodel"
config_path = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(config_path, model_path)

landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=10, min_detection_confidence=0.2)

glasses = cv2.imread("jaron.jpg", cv2.IMREAD_UNCHANGED)  
ring = cv2.imread("800d.png", cv2.IMREAD_UNCHANGED)        

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)


import numpy as np

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
 



while True:
    ret, frame = cap.read()
    if not ret:
        break

    
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
            frame = overlay_transparent(frame, ring, index_x - 15, index_y - 15, ring_size)

    cv2.imshow("Hand and Face + Landmarks + Props", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.00005)

cap.release()
cv2.destroyAllWindows()