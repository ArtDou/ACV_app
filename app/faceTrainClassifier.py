import tkinter as tk
import cv2
import csv
import pickle
import numpy as np
import pandas as pd
from math import hypot
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import mediapipe as mp
from PIL import Image, ImageTk
from glob import glob

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

path_save = "./app/model/dataset/"
pairs = [(145, 159), (80, 88), (13, 14), (374, 386), (310, 318)]
pairwise = (1, 126, 206, 13, 14, 62, 308, 263, 33)
pipelines = {
    "rf": make_pipeline(StandardScaler(), RandomForestClassifier()),
    # 'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
    # 'kn':make_pipeline(StandardScaler(), KNeighborsClassifier()),
}


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


def show_frame():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)


# def capture_images():
#     file_name = glob(path_save + "**", recursive=False)
#     line = liste.curselection()[0]
#     item = liste.get(line)
#     _, frame = cap.read()
#     cv2.imwrite(f"{path_save}{item}_{len(file_name)}.jpeg", frame)


def init_data_file():
    num_coords = len(pairs) + (len(pairwise) ** 2 - len(pairwise)) // 2
    landmarks = ["class"]
    for val in range(1, num_coords + 1):
        landmarks += ["dist{}".format(val)]
    with open("./app/model/dataset/coords.csv", mode="w", newline="") as f:
        csv_writer = csv.writer(
            f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        csv_writer.writerow(landmarks)
    num_coords


def capture_class_100():
    line = liste.curselection()[0]
    item = liste.get(line)
    idx = 0
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        while idx < 100:
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

            if results.multi_face_landmarks:
                # for face_landmarks in results.multi_face_landmarks:
                #     mp_drawing.draw_landmarks(
                #         image=image,
                #         landmark_list=face_landmarks,
                #         connections=mp_face_mesh.FACEMESH_TESSELATION,
                #         landmark_drawing_spec=None,
                #         connection_drawing_spec=mp_drawing_styles
                #         .get_default_face_mesh_tesselation_style())

                try:
                    face_size = max(dist(results, 33, 263), dist(results, 151, 175))
                    face_row = [
                        dist(results, a, b) / face_size * 100 for (a, b) in pairs
                    ]
                    face_row_pairwise = list(
                        pairwise_dist(results, pairwise) / face_size * 100
                    )
                    row = face_row + face_row_pairwise
                    # Append class name
                    row.insert(0, item)
                    # Export to CSV
                    with open(
                        "./app/model/dataset/coords.csv", mode="a", newline=""
                    ) as f:
                        csv_writer = csv.writer(
                            f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
                        )
                        csv_writer.writerow(row)
                    idx += 1
                except Exception as e:
                    print(e)


def capture_class():
    line = liste.curselection()[0]
    item = liste.get(line)
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
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

        if results.multi_face_landmarks:
            # for face_landmarks in results.multi_face_landmarks:
            #     mp_drawing.draw_landmarks(
            #         image=image,
            #         landmark_list=face_landmarks,
            #         connections=mp_face_mesh.FACEMESH_TESSELATION,
            #         landmark_drawing_spec=None,
            #         connection_drawing_spec=mp_drawing_styles
            #         .get_default_face_mesh_tesselation_style())

            try:
                face_size = max(dist(results, 33, 263), dist(results, 151, 175))
                face_row = [dist(results, a, b) / face_size * 100 for (a, b) in pairs]
                face_row_pairwise = list(
                    pairwise_dist(results, pairwise) / face_size * 100
                )
                row = face_row + face_row_pairwise
                # Append class name
                row.insert(0, item)
                # Export to CSV
                with open("./app/model/dataset/coords.csv", mode="a", newline="") as f:
                    csv_writer = csv.writer(
                        f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
                    )
                    csv_writer.writerow(row)
            except Exception as e:
                print(e)


def suppress_class():
    line = liste.curselection()[0]
    item = liste.get(line)
    lines = list()
    classToDelete = item
    with open("./app/model/dataset/coords.csv", "r") as readFile:
        reader = csv.reader(readFile)
        for row in reader:
            lines.append(row)
            for field in row:
                if field == classToDelete:
                    lines.remove(row)
    with open("./app/model/dataset/coords.csv", "w") as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(lines)


def train_model():
    df = pd.read_csv("./app/model/dataset/coords.csv")
    X = df.drop("class", axis=1)  # features
    y = df["class"]  # target value
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    fit_models = {}
    for algo, pipeline in pipelines.items():
        model = pipeline.fit(X_train.values, y_train)
        fit_models[algo] = model
        with open("./app/model/new_model.pkl", "wb") as f:
            pickle.dump(fit_models[algo], f)
    for algo, model in fit_models.items():
        yhat = model.predict(X_test.values)
        print(algo, accuracy_score(y_test, yhat))


width, height = 800, 600
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

root = tk.Tk()
root.bind("<q>", lambda e: root.quit())
# root.bind("<Escape>", lambda e: lmain.pack_forget())
lmain = tk.Label(root)
lmain.pack()

# bouton = tk.Button(root, text="capture image", command=capture_images)
# bouton.pack()
bouton = tk.Button(root, fg="red", text="initialize data file", command=init_data_file)
bouton.pack()
bouton = tk.Button(root, text="capture class x 1", command=capture_class)
bouton.pack()
bouton = tk.Button(root, text="capture class x 100", command=capture_class_100)
bouton.pack()
bouton = tk.Button(root, fg="red", text="suppress class", command=suppress_class)
bouton.pack()
bouton = tk.Button(root, text="train", command=train_model)
bouton.pack()

# liste
liste = tk.Listbox(root)
liste.insert(0, "=(")
liste.insert(1, "=o")
liste.insert(2, "=)")
liste.insert(3, "=D")
liste.insert(4, "?")
liste.pack()

show_frame()

greeting = tk.Label(text="Data Collecter & Model Trainer")
greeting.pack()

root.mainloop()
