import cv2
import numpy as np

def extract_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (150, 150))  # Resize for consistency

    # Calculate color histogram (simple feature)
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist
