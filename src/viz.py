import cv2
import numpy as np
from pathlib import Path

path_to_validation_results = Path('../logs') / '<change-to-name-of-model' / 'preds_per_epoch.npy'
preds_per_epoch = np.load(path_to_validation_results.resolve()).reshape(1, )[0]
frames = preds_per_epoch['frames']
truth = preds_per_epoch['truth']
preds = preds_per_epoch['preds']

use_centers = False

imgs = []
for f, t, p in zip(frames, truth, preds[-1]):
    img = f.copy()
    img = cv2.rectangle(img, (t[0], t[1]), (t[2], t[3]), (255, 255, 255), 2)
    img = cv2.rectangle(img, (p[0], p[1]), (p[2], p[3]), (255, 0, 0), 2)
    img = cv2.rectangle(img, (p[0], p[1]), (p[2], p[3]), (255, 0, 0), 2)
    imgs.append(img)
    cv2.imshow('Tracking', img)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()

height, width, channels = imgs[0].shape
video = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc(*"MJPG"), 60, (width, height))
for img in imgs:
    video.write(img)
video.release()
