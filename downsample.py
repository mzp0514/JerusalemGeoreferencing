import os
import random
import cv2

path = './large_image_cropped'

files = os.listdir(path)
images = []

for f in files:
    if f.split('.')[-1] == 'jpg':
        images.append(path  + '/' + f)

if not os.path.exists('./cropped_data'):
    os.makedirs('./cropped_data')


# random.shuffle(images)
# cnt = 0

for i in range(len(images)):
    img = cv2.imread(images[i])
    root = './large_image_resized'
    res = cv2.resize(img, (1000, 1000), interpolation=cv2.INTER_AREA) 

 
    cv2.imwrite(
        root + '/' + images[i].split('/')[-1].split('.')[0] +
        '.png', res)

