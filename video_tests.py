
import cv2
import numpy as np
import os
from IPython.display import HTML
from base64 import b64encode

def play_video(path):
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Video Playback", frame)

        # Press 'q' to exit
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


play_video("records_cadis_1/file_example_MP4_640_3MG.mp4")