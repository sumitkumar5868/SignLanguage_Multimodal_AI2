import cv2
import csv
import os
import mediapipe as mp


# =========================================================
# SIGNS
# =========================================================

SIGNS = [
    "WATER",
    "FOOD",
    "HELP",
    "BATHROOM",
    "PAIN",
    "MEDICINE",
    "SLEEP",
    "HUNGRY",
    "THIRSTY",
    "EMERGENCY",
    "YES",
    "NO",
    "PLEASE",
    "THANK_YOU",
    "SORRY",
    "HELLO",
    "GOODBYE",
    "GOOD",
    "CALL_DOCTOR",
    "I_AM_FINE"
]

SAMPLES_PER_SIGN = 100

DATASET_FILE = "dataset/custom_sign_dataset.csv"
MODEL_PATH = "model/hand_landmarker.task"


# =========================================================
# CREATE DATASET FOLDER
# =========================================================

os.makedirs("dataset", exist_ok=True)


# =========================================================
# CHECK MODEL
# =========================================================

if not os.path.exists(MODEL_PATH):
    print("ERROR: hand_landmarker.task nahi mila!")
    print("Expected location:")
    print(MODEL_PATH)
    exit()


# =========================================================
# MEDIAPIPE HAND LANDMARKER
# =========================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


# =========================================================
# CSV
# =========================================================

file_exists = os.path.exists(DATASET_FILE)

csv_file = open(
    DATASET_FILE,
    "a",
    newline=""
)

writer = csv.writer(csv_file)


if not file_exists:

    header = []

    for i in range(21):
        header.extend([
            f"x{i}",
            f"y{i}",
            f"z{i}"
        ])

    header.append("label")

    writer.writerow(header)


# =========================================================
# CAMERA
# =========================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Camera open nahi ho raha.")
    csv_file.close()
    exit()


# =========================================================
# VARIABLES
# =========================================================

current_sign = None
count = 0


print()
print("======================================")
print("     CUSTOM SIGN DATASET")
print("======================================")

print()

for i, sign in enumerate(SIGNS, 1):
    print(f"{i}. {sign}")

print()
print("Controls:")
print("1-9 = Select sign")
print("0   = EMERGENCY")
print("SPACE = Capture sample")
print("Q = Quit")
print()


# =========================================================
# HAND LANDMARKER
# =========================================================

with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Camera frame nahi mila.")
            break


        # Mirror camera
        frame = cv2.flip(frame, 1)


        # BGR -> RGB
        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )


        # Detect hand
        result = landmarker.detect(mp_image)


        hand_landmarks = None


        if result.hand_landmarks:

            hand_landmarks = result.hand_landmarks[0]


            # =============================================
            # DRAW 21 LANDMARKS
            # =============================================

            h, w, _ = frame.shape

            for landmark in hand_landmarks:

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )


        # =============================================
        # INFORMATION PANEL
        # =============================================

        cv2.rectangle(
            frame,
            (0, 0),
            (520, 180),
            (0, 0, 0),
            -1
        )


        if current_sign:

            sign_text = current_sign

        else:

            sign_text = "NOT SELECTED"


        cv2.putText(
            frame,
            f"Sign: {sign_text}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Samples: {count}/{SAMPLES_PER_SIGN}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )


        if hand_landmarks:

            status = "HAND DETECTED"
            status_color = (0, 255, 0)

        else:

            status = "SHOW YOUR HAND"
            status_color = (0, 0, 255)


        cv2.putText(
            frame,
            status,
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2
        )


        cv2.putText(
            frame,
            "SPACE = Capture",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )


        cv2.putText(
            frame,
            "Q = Quit",
            (20, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )


        # =============================================
        # SHOW CAMERA
        # =============================================

        cv2.imshow(
            "Custom Sign Language Dataset",
            frame
        )


        # =============================================
        # KEY
        # =============================================

        key = cv2.waitKey(1) & 0xFF


        # =============================================
        # SELECT 1-9
        # =============================================

        if key >= ord("1") and key <= ord("9"):

            index = key - ord("1")

            current_sign = SIGNS[index]

            count = 0

            print()
            print("--------------------------------")
            print(f"Selected Sign: {current_sign}")
            print("Ab hand dikhao.")
            print("SPACE dabakar sample lo.")
            print("--------------------------------")
            print()


        # =============================================
        # SELECT EMERGENCY
        # =============================================

        elif key == ord("0"):

            current_sign = "EMERGENCY"

            count = 0

            print()
            print("Selected Sign: EMERGENCY")
            print()


        # =============================================
        # CAPTURE
        # =============================================

        elif key == 32:

            if current_sign is None:

                print("Pehle sign select karo.")

                continue


            if hand_landmarks is None:

                print("Hand detect nahi hua.")

                continue


            if count >= SAMPLES_PER_SIGN:

                print(
                    f"{current_sign} already complete."
                )

                continue


            # =========================================
            # EXTRACT 21 LANDMARKS
            # =========================================

            features = []

            for landmark in hand_landmarks:

                features.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])


            # Label
            features.append(current_sign)


            # Save CSV
            writer.writerow(features)

            csv_file.flush()


            count += 1


            print(
                f"{current_sign}: "
                f"{count}/{SAMPLES_PER_SIGN}"
            )


            if count == SAMPLES_PER_SIGN:

                print()
                print(
                    f"SUCCESS: {current_sign} complete!"
                )
                print()


        # =============================================
        # QUIT
        # =============================================

        elif key == ord("q"):

            break


# =========================================================
# CLEANUP
# =========================================================

csv_file.close()

cap.release()

cv2.destroyAllWindows()


print()
print("======================================")
print("Dataset collection finished.")
print(f"Saved at: {DATASET_FILE}")
print("======================================")