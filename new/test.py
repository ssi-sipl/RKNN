import cv2

url=""

cap=cv2.VideoCapture(
    url,
    cv2.CAP_FFMPEG
)

while True:

    ret,frame=cap.read()

    if not ret:
        print("lost")
        continue

    cv2.imshow(
        "test",
        cv2.resize(frame,(1280,720))
    )

    if cv2.waitKey(1)==27:
        break