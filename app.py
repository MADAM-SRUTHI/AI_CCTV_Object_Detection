from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")

video = "videos/cctv.mp4"

cap = cv2.VideoCapture(video)

if not cap.isOpened():
    print("Video open avvaledu. cctv.mp4 file check cheyyandi.")
    exit()

log_file = open("detection/detections.txt", "a")

snapshot_saved = False

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)

    for result in results:
        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            object_name = model.names[class_id]

            # Alert System
            if object_name in ["knife", "gun"]:
                print(f"⚠ ALERT: {object_name} detected!")

            log_file.write(
                f"Object: {object_name}, Confidence: {confidence:.2f}\n"
            )

    annotated_frame = results[0].plot()

    # Save first detection image
    if not snapshot_saved and len(results[0].boxes) > 0:
        cv2.imwrite(
            "screenshots/detection.jpg",
            annotated_frame
        )
        snapshot_saved = True

    cv2.imshow(
        "AI CCTV Object Detection",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
log_file.close()
cv2.destroyAllWindows()