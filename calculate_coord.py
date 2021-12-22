import pandas as pd
import numpy as np
import glob
import json
from numpy import linalg as LA

images = glob.glob('./resized_maps/*.png')
images = sorted(images)

data = pd.read_csv('./coordinates.csv')
data2 = pd.read_csv('./subimage_position.csv')

fnames = []
coordinates = []
for i in range(len(images)):
    # loc of points of original image in the subimage coordinate
    position = json.loads(data2['position'][i])

    original_shape = json.loads(data2['original_size'][i])
    cropped_shape = json.loads(data2['cropped_size'][i])

    H, W = original_shape[:2]
    h, w = cropped_shape[:2]

    top_left = [0 - position[0], 0 - position[1]]
    top_right = [0 - position[0], W - position[1]]
    bottom_left = [H - position[0], 0 - position[1]]
    bottom_right = [H - position[0], W - position[1]]

    points = np.array([top_left, top_right, bottom_left, bottom_right])

    (rows, cols) = points.T
    print(data['coordinate'][i])
    vertices = np.array(json.loads(data['coordinate'][i]))
    scale_lat = LA.norm(vertices[0] - vertices[1]) / h
    scale_long = LA.norm(vertices[0] - vertices[2]) / w

    rows = rows * scale_lat
    cols = cols * scale_long

    points = np.array([cols, rows]).T + vertices[3]
    print(points)

    print(np.dot((points[0] - points[1]), points[0] - points[2]),
          LA.norm(points[0] - points[1]), LA.norm(points[0] - points[2]))

    fnames.append(data2['name'][i])
    coordinates.append(points.tolist())

data = pd.DataFrame()
data['name'] = fnames
data['coordinates'] = coordinates
data.to_csv('geo_coordinates.csv')
