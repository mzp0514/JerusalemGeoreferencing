import numpy as np
import cv2
import glob


in_path = './labels'
out_path = './boundary'

files = glob.glob('./processed_images/*_predict.jpg')

for f in files:
    name = f.split('/')[-1]

    img2 = cv2.imread(f)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    print(img2)
    # img2 = 255 * (img2 < 100)
    # img2 = img2.astype('uint8')
    kernel = np.ones((3, 3))
    opening = cv2.morphologyEx(img2, cv2.MORPH_OPEN, kernel)
    
    img2, contours, hierarchy = cv2.findContours(img2, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    c = max(contours, key = cv2.contourArea)
    res = np.zeros(img2.shape)
    cv2.drawContours(res, [c], -1, (255,255,255), 1)

    cv2.imwrite(out_path + '/' + name, res)