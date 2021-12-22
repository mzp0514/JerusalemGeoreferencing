import numpy as np
import cv2
import glob
import os

out_path = './boundaries'

files = glob.glob('./processed_images/*_predict.jpg')

for f in files:
    name = f.split(os.path.sep)[-1]

    img = cv2.imread(f)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # print(img)
    # img = 255 * (img < 100)
    # img = img.astype('uint8')
    kernel = np.ones((3, 3))
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    contours, hierarchy = cv2.findContours(opening,
                                           mode=cv2.RETR_EXTERNAL,
                                           method=cv2.CHAIN_APPROX_NONE)
    c = max(contours, key=cv2.contourArea)
    res = np.zeros(img.shape)
    cv2.drawContours(res, [c], -1, (255, 255, 255), 1)
    print(out_path + '/' + name[:-12] + '.jpg')
    cv2.imwrite(out_path + '/' + name[:-12] + '.jpg', res)
