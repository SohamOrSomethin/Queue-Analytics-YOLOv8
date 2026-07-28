from ultralytics import YOLO
import cv2
import time
import numpy as np

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("video.mp4")

queue_roi = np.array([
    [318, 367],
    [181, 152],
    [174, 107],
    [391,  19],
    [440, 116],
    [381, 150],
    [340, 176],
    [434, 302],
    [451, 330]
], dtype=np.int32)

tracked = {}

while True:

    success, frame = cap.read()

    if not success:
        break

    results = model.track(
    frame,
    persist=True #remember people from the previous frame and assign them an ID, we can track people using this.
    ) #list of result objects is stored in results (FOR EACH FRAME)

    count = 0
    for box in results[0].boxes: #since result is a list of result objects, we access the first result object using results[0] and then access the boxes attribute to get the all bounding boxes for all the detected objects.
    
        current_time = time.time()
        cls = int(box.cls[0]) #class is stored as a tensor which contains exactly one element  hence the cls[0],we convert it to an integer so we can use it for comparison.
        if cls == 0: #class 0 means a person
            conf = float(box.conf[0])
            name = model.names[cls]
            track_id = int(box.id[0])
            label = f"ID:{track_id} {name} {conf:.2f}"
            if box.id is None:
               continue


            x1, y1, x2, y2 = box.xyxy[0] #again a tensor storring coords of top left and bottom right corner of bounding box.

            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0,255,0), #color
                2) #thickness
            
            cv2.putText(
                frame,
                label,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,255,0),
                2
                )
            cv2.polylines(
                frame,
                [queue_roi],
                True,
                (255, 0, 0),
                2
                )
            
            center_x = int((x1 + x2)/2)
            center_y = int((y1 + y2)/2)
            #get center of a detected person and draw a red dot around them then we check if that red dot is innside our defined ROI (queue area), if it is we inc count by 1.
            cv2.circle(
            frame,
            (center_x, center_y),
            4,
            (0,0,255),
            -1
            )

            inside = cv2.pointPolygonTest(
            queue_roi,
            (center_x, center_y),
            False #dont calc dist
            )
            #returns a positive value if the point is inside the polygon, negative if outside, and 0 if on the edge of the polygon.
            if (inside >= 0):
                 if track_id not in tracked:
                  tracked[track_id] = {
                    "enter_time": current_time, 
                    "last_seen": current_time,
                    "counted": False
                    }
                 tracked[track_id]["last_seen"] = current_time
                 #update last seen
                 time_inside = current_time - tracked[track_id]["enter_time"]
                 #calculate the time they spent inside the queue area by current time - time they entered
                 if time_inside >= 3 and tracked[track_id]["counted"] == False: #if they stayed inside for more than 3 secs make counted value true
                   tracked[track_id]["counted"] = True

    for person_id in list(tracked.keys()):
     if current_time - tracked[person_id]["last_seen"] > 2:
        #if a person is not seen for more than 2 seconds, we remove them from the tracked dictionary
        del tracked[person_id]


    count = sum(info["counted"] for info in tracked.values())
    cv2.putText(#we display count once per frame, so we put it outside the for loop
                frame,
                f"Queue Count: {count}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255,255,255),
                2
                )  


    cv2.imshow("People Detection", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()