import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# Canvas
ret, frame = cap.read()
h, w, _ = frame.shape
canvas = np.zeros((h, w, 3), dtype=np.uint8)

prev_x, prev_y = 0, 0

# Blue color range (HSV)
lower_blue = np.array([90, 100, 100])
upper_blue = np.array([130, 255, 255])

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Mask blue fingertip
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)

        if cv2.contourArea(c) > 500:
            (x, y), radius = cv2.minEnclosingCircle(c)
            x, y = int(x), int(y)

            cv2.circle(frame, (x, y), int(radius), (255, 0, 0), 2)

            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = x, y

            cv2.line(canvas, (prev_x, prev_y), (x, y), (255, 0, 255), 5)
            prev_x, prev_y = x, y
    else:
        prev_x, prev_y = 0, 0

    # Merge canvas
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
    inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)

    frame = cv2.bitwise_and(frame, inv)
    frame = cv2.bitwise_or(frame, canvas)

    cv2.putText(frame, "Finger Drawing | C: Clear | Q: Quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)

    cv2.imshow("Finger Air Canvas", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        canvas[:] = 0
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
