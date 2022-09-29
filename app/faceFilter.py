# Les filtres se cumulent
# Appuyez sur '0' pour reset la liste de filtres

import cv2
import mediapipe as mp
from utils.filters import filters, filters_mediapipe

mp_holistic = mp.solutions.holistic

filters_dic = {
    ord("1"): filters.mirror,
    ord("2"): filters.glow,
    ord("3"): filters.sepia,
    ord("4"): filters.b_and_w,
    ord("5"): filters.xRay,
    ord("6"): filters.cartoon,
    ord("7"): filters.drawing,
    ord("8"): filters.thermal_cam,
    ord("9"): filters.quad_cam,
    ord("p"): filters.TV,
    ord("o"): filters.wave,
    ord("n"): filters_mediapipe.red_nose,
    ord("b"): filters_mediapipe.carnival,
    ord("t"): filters.timewarp,
}

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

filter_choice = []
previousFrames = []

with mp_holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
) as holistic:

    while cap.isOpened():

        ret, frame = cap.read()
        if ret:

            previousFrames.append(frame)
            if len(previousFrames) > 21:
                previousFrames.pop(0)

            key = cv2.waitKey(10) & 0xFF

            if key in filters_dic.keys():
                filter_choice.append(filters_dic[key])

            if key == ord("0"):
                filter_choice = []
            else:
                for filter in filter_choice:
                    if not (
                        filter
                        in [
                            filters.timewarp,
                            filters_mediapipe.carnival,
                            filters_mediapipe.red_nose,
                        ]
                    ):
                        frame = filter(frame)
                    elif filter == filters.timewarp:
                        frame = filter(frame, previousFrames)
                    else:
                        frame = filter(holistic, frame)

            cv2.imshow("image", cv2.flip(frame, 1))

            if key == ord("q"):
                break
        else:
            break

cap.release()
cv2.destroyAllWindows()
