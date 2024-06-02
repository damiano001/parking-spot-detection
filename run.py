import cv2
import matplotlib.pyplot as plt
import numpy as np
from common import *
from config import *

mask = cv2.imread(mask_path, 0)
video_capture = cv2.VideoCapture(video_path)
connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
parking_spots = extract_spots(connected_components)

spot_states = [None for _ in parking_spots]
differences = [None for _ in parking_spots]

previous_frame = None

frame_count = 0
frame_step = 30

available_spots_over_time = []
time_stamps = []

fps = video_capture.get(cv2.CAP_PROP_FPS)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    if frame_count % frame_step == 0 and previous_frame is not None:
        for index, spot in enumerate(parking_spots):
            x, y, width, height = spot
            current_crop = frame[y:y + height, x:x + width, :]
            differences[index] = compute_diff(current_crop, previous_frame[y:y + height, x:x + width, :])

        sorted_diffs = [differences[i] for i in np.argsort(differences)][::-1]
        print(sorted_diffs)

    if frame_count % frame_step == 0:
        indices_to_check = range(len(parking_spots)) if previous_frame is None else [i for i in np.argsort(differences) if differences[i] / np.amax(differences) > 0.4]
        for index in indices_to_check:
            x, y, width, height = parking_spots[index]
            current_crop = frame[y:y + height, x:x + width, :]
            spot_status = check(current_crop)
            spot_states[index] = spot_status

    if frame_count % frame_step == 0:
        previous_frame = frame.copy()

    for index, spot in enumerate(parking_spots):
        x, y, width, height = spot
        color = (0, 255, 0) if spot_states[index] else (0, 0, 255)
        frame = cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
        cv2.putText(frame, f'{index + 1}', (x + 5, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)  

    total_spots = len(spot_states)
    available_spots = sum(spot_states)
    available_percentage = (available_spots / total_spots) * 100

    
    text = f'Available spots: {available_spots} of {total_spots} ({available_percentage:.2f}%)'
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 1.2, 3)
    cv2.rectangle(frame, (80, 20), (80 + text_width + 20, 20 + text_height + 20), (255, 255, 255), -1)  
    cv2.putText(frame, text, (90, 20 + text_height + 10), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 0), 3)  
    cv2.namedWindow('Parking Lot', cv2.WINDOW_NORMAL)
    cv2.imshow('Parking Lot', frame)

    current_time = frame_count / fps
    available_spots_over_time.append(available_spots)
    time_stamps.append(current_time)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    frame_count += 1

video_capture.release()
cv2.destroyAllWindows()


plt.plot(time_stamps, available_spots_over_time)
plt.xlabel('Time (seconds)')
plt.ylabel('Available Spots')
plt.title('Available Spots Over Time')
plt.grid(True)
plt.show()
















