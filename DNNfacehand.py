#libs
import cv2
import dlib
import time
import mediapipe as mp

#load all models

#opencv face detection using ssd dnn model
model_path = "res10_300x300_ssd_iter_140000.caffemodel"
config_path = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(config_path, model_path)
#dlib landmark detection using random github dnn model
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#mediapipe hands detection usng dnn
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=10, min_detection_confidence=0.2)

#video feed
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)

#face and landmark while loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    #flip the frame
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    #face detect (scale factor for size variation)
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        #confidence threshold for rectangle
        if confidence > 0.4:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (startX, startY, endX, endY) = box.astype("int")

            #confidence text and rectangle outline
            text = f"Confidence: {confidence:.2f}"
            cv2.putText(frame, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
            face_roi = dlib.rectangle(startX, startY, endX, endY)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            landmarks = landmark_predictor(gray, face_roi)

        #confidence threshold for landmarks
        if confidence > 0.75:
            for n in range(68):
                x, y = landmarks.part(n).x, landmarks.part(n).y
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    #hand detection (mp)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    #hand detection
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (255, 255, 0), -1)

            #draw on the video feed
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2)
            )

    cv2.imshow("hand and face + landmarks", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.00005)

cap.release()
cv2.destroyAllWindows()