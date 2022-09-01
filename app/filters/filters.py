import numpy as np
import cv2
import math
# Fonctions de filtre image

def mirror(img):
    width = img.shape[1]
    flipVertical = np.fliplr(img)
    final = sidebyside(img[:,:width//2,:],flipVertical[:,width//2:,:])
    return final

def glow(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    v = cv2.multiply(v,1.2)
    v = cv2.GaussianBlur(v,(5,5),1)
    merged = cv2.merge([h, s, v])
    final = cv2.cvtColor(merged, cv2.COLOR_HSV2BGR)
    return final

def sepia(img):
    img = np.array(img, dtype=np.float64) # converting to float to prevent loss
    img = cv2.transform(img, np.matrix([[0.272, 0.534, 0.131],
                                        [0.349, 0.686, 0.168],
                                        [0.393, 0.769, 0.189]])) # multipying image with special sepia matrix
    img[np.where(img > 255)] = 255 # normalizing values greater than 255 to 255
    img = np.array(img, dtype=np.uint8) # converting back to int
    return img

def b_and_w(img):
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
    return blackAndWhiteImage

def xRay(img):
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    frayInv = 255-grayImage
    xRay = cv2.applyColorMap(frayInv, cv2.COLORMAP_OCEAN)
    return xRay

def cartoon(img):
    tublur = cv2.medianBlur(img, 5)
    # We'll cover Canny edge detection and dilation shortly
    edge = cv2.Canny(tublur, 100, 200)
    kernel = np.ones((3,3), np.uint8)
    edge = cv2.dilate(edge, kernel, iterations = 1)
    tublur[edge==255] = 0
    return tublur

def drawing(img):
    pencil_gray, pencil_color = cv2.pencilSketch(img, sigma_s=100, sigma_r=0.07, shade_factor=0.05) # inbuilt function to generate pencil sketch in both color and grayscale
    return pencil_gray

def thermal_cam(img):
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thermal = cv2.applyColorMap(grayImage, cv2.COLORMAP_JET)
    return thermal

def TV(img):
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = 0.1 # creating threshold. This means noise will be added to 80% pixels
    for i in range(height):
        for j in range(width):
            if np.random.rand() <= thresh:
                if np.random.randint(2) == 0:
                    gray[i, j] = min(gray[i, j] + np.random.randint(0, 64), 255) # adding random value between 0 to 64. Anything above 255 is set to 255.
                else:
                    gray[i, j] = max(gray[i, j] - np.random.randint(0, 64), 0) # subtracting random values between 0 to 64. Anything below 0 is set to 0.
    return gray

def quad_cam(img):
    resized = cv2.resize(img,None,fx=0.5,fy=0.5, interpolation = cv2.INTER_LINEAR)
    merged = sidebyside_four(resized,resized,resized,resized)
    return merged

def wave(img):

    img_output = np.zeros(img.shape, dtype=img.dtype)
    rows , cols, _= img.shape

    for i in range(rows):
        for j in range(cols):
            offset_x = int(20.0 * math.sin(2 * 3.14 * i / 150))
            offset_y = int(20.0 * math.cos(2 * 3.14 * j / 150))
            if i+offset_y < rows and j+offset_x < cols:
                img_output[i,j] = img[(i+offset_y)%rows,(j+offset_x)%cols]
            else:
                img_output[i,j] = 0

    return img_output   

def sidebyside_four(imageleftup, imagerightup, imageleftdown, imagerightdown):
    newimheight = max(imageleftup.shape[0], imagerightup.shape[0],imageleftdown.shape[0],imagerightdown.shape[0])*2
    newimwidth = max(imageleftup.shape[1], imagerightup.shape[1], imageleftdown.shape[1], imagerightdown.shape[1])*2
    
    newim = np.zeros((newimheight, newimwidth, 3), dtype=np.uint8)

    newim[:imageleftup.shape[0], :imageleftup.shape[1]] = imageleftup
    newim[:imagerightup.shape[0], -imagerightup.shape[1]:] = imagerightup
    newim[-imageleftdown.shape[0]:, :imageleftdown.shape[1]] = imageleftdown
    newim[-imagerightdown.shape[0]:, -imagerightdown.shape[1]:] = imagerightdown

    return newim

def sidebyside(imageleft, imageright):
    newimheight = max(imageleft.shape[0], imageright.shape[0])
    newimwidth = imageleft.shape[1] + imageright.shape[1]
    
    newim = np.zeros((newimheight, newimwidth, 3), dtype=np.uint8)

    newim[:imageleft.shape[0], :imageleft.shape[1]] = imageleft
    newim[:imageright.shape[0], -imageright.shape[1]:] = imageright

    return newim

def timewarp(img,previousFr):
    try:
        b_0,g,r = cv2.split(previousFr[4])
        b,g_20,r = cv2.split(previousFr[12])
        b,g,r_40 = cv2.split(previousFr[20])
        img = cv2.merge([b_0,g_20,r_40])
        # blended = cv2.addWeighted(frame,0.25,merged,0.75,0)
    except:
        pass
    return img