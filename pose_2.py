import cv2
import mediapipe as mp
import numpy as np

# Load t-shirt image with alpha channel
shirt = cv2.imread("input.png", cv2.IMREAD_UNCHANGED)
if shirt is None or shirt.shape[2] != 4:
    print("Error: input.png not found or missing alpha channel.", shirt.shape[2])
    exit()
print("Loaded shirt image shape:", shirt.shape)
print("Shirt alpha max:", np.max(shirt[:, :, 3]))
# Initialize camera
cap = cv2.VideoCapture(0)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


def rotate_image_with_alpha(image, angle):
    """Rotates an RGBA image around its top-center with full alpha preservation"""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # Calculate bounding box size after rotation
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust rotation matrix to take translation into account
    M[0, 2] += (new_w // 2) - center[0]
    M[1, 2] += (new_h // 2) - center[1]

    # Perform affine transformation
    rotated = cv2.warpAffine(
        image,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),  # Transparent background
    )

    # Ensure output has 4 channels
    if rotated.shape[2] != 4:
        alpha = np.ones((rotated.shape[0], rotated.shape[1], 1), dtype=np.uint8) * 255
        rotated = np.concatenate((rotated, alpha), axis=2)

    return rotated


def overlay_transparent(bg, overlay, x, y):
    bh, bw = bg.shape[:2]
    h, w = overlay.shape[:2]
    # Ensure overlay does not go out of bounds
    if x >= bw or y >= bh or x + w <= 0 or y + h <= 0:
        return bg
    # Clip overlay region if needed
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, bw), min(y + h, bh)

    src_x0 = max(0, -x)
    src_y0 = max(0, -y)
    dst_x0 = max(0, x)
    dst_y0 = max(0, y)

    # width / height we can actually draw
    w_draw = min(w - src_x0, bw - dst_x0)
    h_draw = min(h - src_y0, bh - dst_y0)
    overlay_roi = overlay[src_y0 : src_y0 + h_draw, src_x0 : src_x0 + w_draw]
    bg_roi = bg[dst_y0 : dst_y0 + h_draw, dst_x0 : dst_x0 + w_draw]
    # Separate alpha channel and normalize
    alpha = overlay_roi[:, :, 3] / 255.0
    alpha_inv = 1.0 - alpha
    for c in range(3):  # BGR
        bg_roi[:, :, c] = alpha * overlay_roi[:, :, c] + alpha_inv * bg_roi[:, :, c]
    bg[dst_y0 : dst_y0 + h_draw, dst_x0 : dst_x0 + w_draw] = bg_roi

    return bg


while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    mp_drawing = mp.solutions.drawing_utils

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )
        landmarks = results.pose_landmarks.landmark
        h, w, _ = frame.shape

        l_sh = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_sh = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        l_hip, r_hip = (
            landmarks[mp_pose.PoseLandmark.LEFT_HIP],
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP],
        )
        # Get pixel positions
        x1, y1 = int(l_sh.x * w), int(l_sh.y * h)
        x2, y2 = int(r_sh.x * w), int(r_sh.y * h)
        shoulder_dist = int(np.hypot(x2 - x1, y2 - y1))
        hip_y = int((l_hip.y + r_hip.y) / 2 * h)
        neck_y = int((y1 + y2) / 2)
        torso_len = hip_y - neck_y
        shirt_width = int(1.5 * shoulder_dist)
        shirt_height = int(1.5 * torso_len)

        if shirt_width < 20 or shirt_height < 20:
            continue

        # Compute center, width, angle
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        # Resize shirt
        resized_shirt = cv2.resize(shirt, (shirt_width, shirt_height))

        # Rotate Shirt
        rotated_shirt = rotate_image_with_alpha(resized_shirt, -angle + 180)

        if rotated_shirt.shape[2] != 4:
            alpha = (
                np.ones(
                    (rotated_shirt.shape[0], rotated_shirt.shape[1], 1), dtype=np.uint8
                )
                * 255
            )
            rotated_shirt = np.concatenate((rotated_shirt, alpha), axis=2)

        # Adjust overlay position
        top_left_x = center_x - shirt_width // 2
        top_left_y = neck_y - shirt_height // 4

        # Overlay on frame
        print("Left shoulder:", x1, y1, "Right shoulder:", x2, y2)
        print("Shirt size:", shirt_width, shirt_height)
        print("Overlay at:", top_left_x, top_left_y)

        frame = overlay_transparent(frame, rotated_shirt, top_left_x, top_left_y)
        # test_overlay = cv2.resize(shirt, (300, 300))
        # frame = overlay_transparent(frame, test_overlay, 100, 100)

        cv2.putText(
            frame,
            "Shirt overlay applied",
            (30, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    cv2.imshow("Virtual Trial Room", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
