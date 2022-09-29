import cv2
import pickle
import numpy as np
import pandas as pd
from math import hypot
import mediapipe as mp
from utils.classifier import dist, pairwise_dist
from glob import glob

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

face_class = "NA"
face_prob = [0]
# Set which landmarks/pairs of landmark to use to train your algorithm
pairs = [(145, 159), (80, 88), (13, 14), (374, 386), (310, 318)]
pairwise = (1, 126, 206, 13, 14, 62, 308, 263, 33)

if "./app/model/new_model.pkl" in glob("./app/model/**"):
    model_used = "new_model"
else:
    model_used = "default_model"
print(model_used)

with open(f"./app/model/{model_used}.pkl", "rb") as f:
    model = pickle.load(f)

idx = 0
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = face_mesh.process(image)
        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_height, image_width, _ = image.shape

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                )

            if idx % 5 == 0:
                face_size = max(dist(results, 33, 263), dist(results, 151, 175))
                face_row = [dist(results, a, b) / face_size * 100 for (a, b) in pairs]
                face_row_pairwise = list(
                    pairwise_dist(results, pairwise) / face_size * 100
                )
                row = face_row + face_row_pairwise
                X = pd.DataFrame([row])
                face_class = model.predict(X)[0]
                face_prob = model.predict_proba(X)[0]
                # print(face_class, face_prob)

            # Get status box
            cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)

            # Display Class
            cv2.putText(
                image,
                "CLASS",
                (95, 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                face_class,
                (90, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Display Probability
            cv2.putText(
                image,
                "PROB",
                (15, 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                str(round(face_prob[np.argmax(face_prob)], 2)),
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("Mood Detector", image)
        idx += 1
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
cap.release()
cv2.destroyAllWindows()
