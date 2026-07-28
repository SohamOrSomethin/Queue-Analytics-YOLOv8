import cv2
import numpy as np

cap = cv2.VideoCapture("video.mp4")
ret, img = cap.read()
cap.release()
points = []

def mouse_click(event, x, y, flags, param):
    global img

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

        # Draw point
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

        # Draw line to previous point
        if len(points) > 1:
            cv2.line(img, points[-2], points[-1], (255, 0, 0), 2) #2nd last point to last point line banao

        cv2.imshow("Select Queue ROI", img)

cv2.namedWindow("Select Queue ROI")
cv2.setMouseCallback("Select Queue ROI", mouse_click)

while True:

    cv2.imshow("Select Queue ROI", img)

    key = cv2.waitKey(1)

    # Press C to close polygon
    if key == ord('c'):

        if len(points) > 2:
            cv2.line(img, points[-1], points[0], (255,0,0), 2)
            cv2.imshow("Select Queue ROI", img)

    # Press S to save coordinates
    elif key == ord('s'):
        break

cv2.destroyAllWindows()

print("\nPolygon Coordinates:\n")
print(np.array(points, dtype=np.int32))