import cvzone
from cvzone.PoseModule import PoseDetector
import cv2
import os
import numpy as np


def video_pose():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Camera did not open.")
        exit()

    actual_chest_in = 46.5
    actual_shoulder_in = 22.1
    actual_length_in = 28.9

    print("Camera opened. Waiting for frames...")

    detector = PoseDetector(staticMode=False,
                            modelComplexity=1,
                            smoothLandmarks=True,
                            enableSegmentation=False,
                            smoothSegmentation=True,
                            detectionCon=0.5,
                            trackCon=0.5)
    shirtFolderPath = "Resources/Shirts"
    listShirts = os.listdir(shirtFolderPath)
    from PIL import Image
    i = Image.open(os.path.join(shirtFolderPath, listShirts[0]))
    print(i.size[0], i.size[1])
    shirtRatioHeightWidth = i.size[1]/i.size[0]
    while True:
        success, img = cap.read()
        img = detector.findPose(img, draw=False)

        lmList, bboxInfo = detector.findPosition(img, draw=False, bboxWithHands=False)

        if lmList:
            imgShirt = cv2.imread(os.path.join(shirtFolderPath, listShirts[0]), cv2.IMREAD_UNCHANGED)


            lm11 = lmList[11][0:2]
            lm12 = lmList[12][0:2]

            lm23 = lmList[23][0:2]
            lm24 = lmList[24][0:2]

            detected_shoulder_px = lm11[0] - lm12[0]

            pixels_per_inch = detected_shoulder_px / actual_shoulder_in
            length_px = actual_length_in * pixels_per_inch

            scale = (lm11[0]-lm12[0])/500
            wMul = 100 * scale
            hMul = 150 * scale

            lm11 = [lm11[0] + wMul, lm11[1] - hMul]
            lm12 = [lm12[0] - wMul, lm12[1] - hMul]
            lm23 = [lm23[0] + wMul * 2.5, lm23[1]]
            lm24 = [lm24[0] - wMul * 2.5, lm24[1]]

            h, w, c = imgShirt.shape
            src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
            dest = np.array([lm11, lm12, lm24, lm23], dtype=np.float32)

            matrix = cv2.getPerspectiveTransform(src, dest)
            imgShirt = cv2.warpPerspective(imgShirt, matrix, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_TRANSPARENT)
            try:
                img = cvzone.overlayPNG(img, imgShirt, (0, 0))
            except:
                pass

        # Display the frame in a window
        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
def image_pose(img_path):
    img = cv2.imread(img_path)

    # Initialize Pose Detector
    detector = PoseDetector(staticMode=True,  # Static mode since we're using an image
                            modelComplexity=1,
                            smoothLandmarks=True,
                            enableSegmentation=False,
                            smoothSegmentation=True,
                            detectionCon=0.5,
                            trackCon=0.5)

    # Detect pose in the image
    img = detector.findPose(img, draw=True)  # Draw the full pose on the image
    lmList, bboxInfo = detector.findPosition(img, draw=False, bboxWithHands=False)

    if lmList:
        lm11 = lmList[11][0:2]
        lm12 = lmList[12][0:2]
        lm23 = lmList[23][0:2]
        lm24 = lmList[24][0:2]
        cv2.imshow("Pose Detection", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return "yes"
    else:
        return None, None
    
print(image_pose("person.jpg"))

