import cv2
import dlib
import mediapipe as mp
import datetime
import time
import numpy as np
import imutils

modelPath = "res10_300x300_ssd_iter_140000.caffemodel"
configPath = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configPath, modelPath)
landmarkPredictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=10, min_detection_confidence=0.2)

lastCaptureTime = 0
captureCooldown = 2

glasses = cv2.resize(cv2.imread("assets/head1.png", cv2.IMREAD_UNCHANGED),(500,500))
ring = cv2.resize(cv2.imread("assets/hand1.png", cv2.IMREAD_UNCHANGED),(200, 200))
ring = imutils.rotate(ring, 0)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)

def overlayTransparent(bg, overlayImg, x, y, overlaySize=None):
    if overlayImg is None or overlayImg.shape[0] == 0 or overlayImg.shape[1] == 0:
        print("no prop image")
        return bg

    if overlaySize is not None:
        try:
            overlayImg = cv2.resize(overlayImg, overlaySize, interpolation=cv2.INTER_AREA)
        except ValueError:
            pass

    h, w = overlayImg.shape[:2]
    if x + w > bg.shape[1] or y + h > bg.shape[0]:
        return bg

    if overlayImg.shape[2] == 3:
        b, g, r = cv2.split(overlayImg)
        alpha = np.ones(b.shape, dtype=b.dtype) * 255
        overlayImg = cv2.merge((b, g, r, alpha))

    overlayImage = overlayImg[:, :, :3]
    mask = overlayImg[:, :, 3:] / 255.0

    roi = bg[y:y+h, x:x+w]

    if roi.shape[:2] != mask.shape[:2]:
        print("size mismatch")
        return bg

    bg[y:y+h, x:x+w] = (1 - mask) * roi + mask * overlayImage
    return bg

def overlayTransparentPerspective(bg, overlayImg, pts1, pts2):
    if overlayImg is None or overlayImg.shape[0] == 0 or overlayImg.shape[1] == 0:
        print("No prop image")
        return bg

    M = cv2.getPerspectiveTransform(np.float32(pts1), np.float32(pts2))
    h, w = bg.shape[:2]
    overlayWarped = cv2.warpPerspective(overlayImg, M, (w, h))

    mask = overlayWarped[:, :, 3:] / 255.0
    overlayImage = overlayWarped[:, :, :3]

    roi = bg[:overlayWarped.shape[0], :overlayWarped.shape[1]]
    if roi.shape[:2] != mask.shape[:2]:
        print("Size mismatch")
        return bg

    bg[:overlayWarped.shape[0], :overlayWarped.shape[1]] = (1 - mask) * roi + mask * overlayImage
    return bg

def palmClose(handLandmarks):
    fingers = [
        (mpHands.HandLandmark.INDEX_FINGER_TIP, mpHands.HandLandmark.INDEX_FINGER_MCP),
        (mpHands.HandLandmark.MIDDLE_FINGER_TIP, mpHands.HandLandmark.MIDDLE_FINGER_MCP),
        (mpHands.HandLandmark.RING_FINGER_TIP, mpHands.HandLandmark.RING_FINGER_MCP),
        (mpHands.HandLandmark.PINKY_TIP, mpHands.HandLandmark.PINKY_MCP)
    ]
    return all(handLandmarks.landmark[tip].y >= handLandmarks.landmark[mcp].y for tip, mcp in fingers)

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

            faceRoi = dlib.rectangle(startX, startY, endX, endY)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            landmarks = landmarkPredictor(gray, faceRoi)

        if confidence > 0.60:
            for n in range(68):
                xVal, yVal = landmarks.part(n).x, landmarks.part(n).y

            pts1 = np.array([
                [0, 0],
                [glasses.shape[1], 0],
                [0, glasses.shape[1]],
                [glasses.shape[1], glasses.shape[1]]
            ], dtype=np.float32)

            eyeLeft = np.array([landmarks.part(36).x, landmarks.part(36).y])
            eyeRight = np.array([landmarks.part(45).x, landmarks.part(45).y])

            dx = eyeRight[0] - eyeLeft[0]
            dy = eyeRight[1] - eyeLeft[1]
            angle = np.arctan2(dy, dx)

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

            frame = overlayTransparentPerspective(frame, glassesResized, pts1, pts2)

    rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgbFrame)

    if result.multi_hand_landmarks:
        for handLandmarks in result.multi_hand_landmarks:
            wrist = handLandmarks.landmark[mpHands.HandLandmark.WRIST]
            middleMcp = handLandmarks.landmark[mpHands.HandLandmark.MIDDLE_FINGER_MCP]
            indexMcp = handLandmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_MCP]
            pinkyMcp = handLandmarks.landmark[mpHands.HandLandmark.PINKY_MCP]

            wristX, wristY = int(wrist.x * w), int(wrist.y * h)
            middleX, middleY = int(middleMcp.x * w), int(middleMcp.y * h)
            indexX, indexY = int(indexMcp.x * w), int(indexMcp.y * h)
            pinkyX, pinkyY = int(pinkyMcp.x * w), int(pinkyMcp.y * h)

            pts1 = np.array([
                [0, 0],
                [ring.shape[1], 0],
                [0, ring.shape[0]],
                [ring.shape[1], ring.shape[0]]
            ], dtype=np.float32)

            indexPip = handLandmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_PIP]
            pinkyPip = handLandmarks.landmark[mpHands.HandLandmark.PINKY_PIP]

            indexPipX, indexPipY = int(indexPip.x * w), int(indexPip.y * h)
            pinkyPipX, pinkyPipY = int(pinkyPip.x * w), int(pinkyPip.y * h)

            widthTop = np.linalg.norm(np.array([indexX, indexY]) - np.array([pinkyX, pinkyY]))
            widthBottom = np.linalg.norm(np.array([indexPipX, indexPipY]) - np.array([pinkyPipX, pinkyPipY]))
            avgWidth = (widthTop + widthBottom) / 2

            heightLeft = np.linalg.norm(np.array([indexX, indexY]) - np.array([indexPipX, indexPipY]))
            heightRight = np.linalg.norm(np.array([pinkyX, pinkyY]) - np.array([pinkyPipX, pinkyPipY]))
            avgHeight = (heightLeft + heightRight) / 2

            pts2 = np.array([
                [indexX, indexY],
                [pinkyX, pinkyY],
                [indexX, indexY + int(avgHeight)],
                [pinkyX, pinkyY + int(avgHeight)]
            ], dtype=np.float32)

            propAspect = ring.shape[1] / ring.shape[0]
            handWidth = np.linalg.norm(np.array([indexX, indexY]) - np.array([pinkyX, pinkyY]))
            handHeight = np.linalg.norm(np.array([wristX, wristY]) - np.array([middleX, middleY]))

            scaleFactor = handWidth / ring.shape[1]
            if handHeight / handWidth > propAspect:
                scaleFactor = handHeight / ring.shape[0]

            sizeMult = 2

            scaledWidth = int(ring.shape[1] * scaleFactor * sizeMult)
            scaledHeight = int(ring.shape[0] * scaleFactor * sizeMult)

            ringResized = cv2.resize(ring, (scaledWidth, scaledHeight))

            frame = overlayTransparentPerspective(frame, ringResized, pts1, pts2)

            currTime = time.time()
            if palmClose(handLandmarks) and (currTime - lastCaptureTime > captureCooldown):
                lastCaptureTime = currTime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"capture_{timestamp}.png"
                cv2.imwrite(filename, frame)
                print(f"SS saved as {filename}")

                flash = np.ones_like(frame, dtype=np.uint8) * 255
                alphaVal = 0.75
                flashFrame = cv2.addWeighted(frame, 1 - alphaVal, flash, alphaVal, 0)
                cv2.imshow("JARON SIMULATOR", flashFrame)
                cv2.waitKey(200)

    cv2.imshow("JARON SIMULATOR", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
