import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera open nahi hua")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame nahi mil raha")
        break

    # Hand ke liye baad mein yahin MediaPipe lagayenge
    cv2.imshow("Live Camera - Sign Language", frame)

    # Q press karke camera band karo
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()