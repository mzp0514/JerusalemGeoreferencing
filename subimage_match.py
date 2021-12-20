import os
import cv2
import json
import glob
import pandas as pd
import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt
import aircv as ac

path = './raw_data'
subimage_path = './data2'
finalimage_path = './cropped_data'

images = glob.glob('./raw_data/*.jpg')

images = sorted(images)

data = pd.read_csv('coordinates.csv') 


fnames = []
positions = []
original_size = []
cropped_size = []
for i in range(len(images)):
    img_name = images[i].split('\\')[-1].split('.')[0]
    img = ac.imread(images[i])
    sub_img = ac.imread(subimage_path + '/' + img_name + '.jpg')
    final_img = cv2.imread(finalimage_path + '/' + img_name + '.png')
    # print(img_name)
    H, W = img.shape[0], img.shape[1]
    h, w = sub_img.shape[0], sub_img.shape[1]
    h_final, w_final = final_img.shape[0], final_img.shape[1]

    # res = cv2.matchTemplate(img, sub_img, cv2.TM_CCOEFF)
    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    pos = ac.find_template(img, sub_img)

    if pos == None:
        print(img_name)
        continue

    # print(pos['rectangle'])
    
    coord = pos['rectangle'][0]

    

    ## loc of points of original image in the subimage coordinate
    
    # top_left = [0 - max_loc[0], 0 - max_loc[1]] 
    # top_right = [0 - max_loc[0], W - max_loc[1]] 
    # bottom_left = [H - max_loc[0], 0 - max_loc[1]] 
    # bottom_right = [H - max_loc[0], W - max_loc[1]] 

    # points = np.array([top_left, top_right, bottom_left, bottom_right])

    # (rows, cols) = points.T
    # print(data['coordinate'][i])
    # vertices = np.array(json.loads(data['coordinate'][i]))
    # scale_lat = LA.norm(vertices[0] - vertices[1]) / h
    # scale_long = LA.norm(vertices[0] - vertices[2]) / w

    # rows = rows * scale_lat
    # cols = cols * scale_long

    # points = np.array([cols, rows]).T

    # points = points + vertices[3]

    fnames.append(img_name)
    positions.append([coord[0], coord[1]])
    original_size.append([H, W])
    cropped_size.append([h, w])

    # assert coord[0] + h < H and coord[1] + w < W

    print(coord, [H, W], [h, w])
    
    # break

    # cv2.rectangle(img, coord, (coord[0] + h, coord[1] + w), 255, 2)

    # plt.subplot(121),plt.imshow(res,cmap = 'gray')
    # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(img,cmap = 'gray')
    # plt.title('Detected Point'), plt.xticks([]), plt.yticks([])

    # plt.show()

    

data = pd.DataFrame()
data['name'] = fnames
data['position'] = positions
data['original_size'] = original_size
data['cropped_size'] = cropped_size
data.to_csv('coordinates2.csv')  



