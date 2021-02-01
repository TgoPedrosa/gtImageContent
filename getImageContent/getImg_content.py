# USAGE
# python getImg_content.py  --image digital_teste.bmp
# author: Tiago Pedrosa
import numpy as np
import cv2
import argparse



# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False,
	help="Imagem que será alinhada com o template")
args = vars(ap.parse_args())



def cropImage(img):
    
    rgb = img.copy()
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 10))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    # using RETR_EXTERNAL instead of RETR_CCOMP
    contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.zeros(bw.shape, dtype=np.uint8)

    for idx in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx])
        mask[y:y+h, x:x+w] = 0
        cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
        r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)

        if r > 0.45 and w > 50:
            cv2.rectangle(rgb, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)
            crop = rgb[y:y+h, x:x+w]
    
    cv2.imshow("Matched Keypoints", img)
    cv2.waitKey(0)
    cv2.imshow("Matched Keypoints", connected)
    cv2.waitKey(0)
    cv2.imshow("Matched Keypoints", rgb)
    cv2.waitKey(0)
    cv2.imshow("Matched Keypoints", crop)
    cv2.waitKey(0)


input_img = cv2.imread(args["image"])
final_image = cropImage(input_img)  
