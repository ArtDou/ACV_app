import cv2
import mediapipe as mp
import numpy as np
mp_holistic = mp.solutions.holistic

def mediapipe_result(image):
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
    return results

def red_nose(img):
    try:  
        results = mediapipe_result(img) 
        nose = results.face_landmarks.landmark[1]
        img = cv2.circle(img, (int(nose.x*img.shape[1]), int(nose.y*img.shape[0])) , int((-400*nose.z+1)), (0, 0, 255), -1)
    except:
        pass
    return img

def carnival(img):
    img2_import = cv2.imread("filters/data/carniva.png",cv2.IMREAD_UNCHANGED)
    img2_import = cv2.resize(img2_import, None,fx=0.06, fy=0.06, interpolation = cv2.INTER_LINEAR)
    img2 = img2_import[:,:,:3]
    alpha = np.array([img2_import[:,:,3],img2_import[:,:,3],img2_import[:,:,3]],dtype='uint8')
    alpha = np.moveaxis(alpha,0,-1)

    try:  
        results = mediapipe_result(img) 
        ear_l = results.face_landmarks.landmark[71]
        ear_l_coord = [int(ear_l.y*img.shape[0]), int(ear_l.x*img.shape[1])]
        ear_r = results.face_landmarks.landmark[301]
        ear_r_coord = [int(ear_r.y*img.shape[0]), int(ear_r.x*img.shape[1])]
        nose_up = results.face_landmarks.landmark[94]
        nose_up_coord = [int(nose_up.y*img.shape[0]), int(nose_up.x*img.shape[1])]

        img1 = img.copy()

        img_fake = np.zeros(img1.shape, dtype='uint8')
        mask_fake = np.zeros(img1.shape, dtype='uint8')

        img_fake[:img2.shape[0],:img2.shape[1],:]=img2
        mask_fake[:img2.shape[0],:img2.shape[1],:]=alpha

        pts2 = np.float32([ear_l_coord,nose_up_coord,ear_r_coord])[:,::-1]
        pts1 = np.float32([[24,22],[92,89],[24,156]])[:,::-1]
        M = cv2.getAffineTransform(pts1,pts2)
        img_fake = cv2.warpAffine(img_fake, M,(img_fake.shape[1],img_fake.shape[0]))
        mask_fake = cv2.warpAffine(mask_fake,M,(mask_fake.shape[1],mask_fake.shape[0]))

        img_fake = img_fake.astype('float')
        mask_fake = mask_fake.astype('float') / 255

        img = (mask_fake * img_fake + (1. - mask_fake)*img1).astype('uint8')
        
    except:
        pass
    return img
