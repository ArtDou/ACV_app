import csv
import cv2
import numpy as np
from math import hypot
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def dist(res, a, b):
    up = res.multi_face_landmarks[0].landmark[a]
    down = res.multi_face_landmarks[0].landmark[b]
    return hypot(up.x - down.x, up.y - down.y)


def pairwise_dist(res, list_points):
    X = [
        [
            res.multi_face_landmarks[0].landmark[a].x,
            res.multi_face_landmarks[0].landmark[a].y,
        ]
        for a in list_points
    ]
    a = pairwise_distances(X, n_jobs=-1)
    return a[np.triu_indices(len(a), k=1)]


def create_file(num_coords, path):
    landmarks = ["class"]
    for val in range(1, num_coords + 1):
        landmarks += ["dist{}".format(val)]
    with open(path, mode="w", newline="") as f:
        csv_writer = csv.writer(
            f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        csv_writer.writerow(landmarks)
    num_coords
