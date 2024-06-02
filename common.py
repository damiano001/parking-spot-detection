from skimage.transform import resize
import numpy as np
import cv2
from config import *

def compute_diff(image1, image2):
    return np.abs(np.mean(image1) - np.mean(image2))


def check(spot_image):
    flattened_data = []
    resized_image = resize(spot_image, (15, 15, 3))
    flattened_data.append(resized_image.flatten())
    flattened_data = np.array(flattened_data)
    prediction = MODEL.predict(flattened_data)

    return SPOT_EMPTY if prediction == 0 else SPOT_NOT_EMPTY

def extract_spots(connected_components):
    (num_labels, label_ids, stats, centroids) = connected_components

    parking_slots = []
    scale_factor = 1

    for i in range(1, num_labels):
        x1 = int(stats[i, cv2.CC_STAT_LEFT] * scale_factor)
        y1 = int(stats[i, cv2.CC_STAT_TOP] * scale_factor)
        width = int(stats[i, cv2.CC_STAT_WIDTH] * scale_factor)
        height = int(stats[i, cv2.CC_STAT_HEIGHT] * scale_factor)

        parking_slots.append([x1, y1, width, height])

    return parking_slots

