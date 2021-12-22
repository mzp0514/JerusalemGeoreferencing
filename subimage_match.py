import os
import glob
import pandas as pd
import aircv as ac

subimage_path = './cropped_maps'
images = glob.glob('./raw_maps/*.jpg')

images = sorted(images)

fnames = []
positions = []
original_size = []
cropped_size = []
for i in range(len(images)):
    img_name = images[i].split(os.path.sep)[-1].split('.')[0]
    img = ac.imread(images[i])
    sub_img = ac.imread(subimage_path + os.path.sep + img_name + '.jpg')

    # print(img_name)
    H, W = img.shape[0], img.shape[1]
    h, w = sub_img.shape[0], sub_img.shape[1]

    pos = ac.find_template(img, sub_img)

    if pos == None:
        print(img_name)
        continue

    coord = pos['rectangle'][0]

    fnames.append(img_name)
    positions.append([coord[0], coord[1]])
    original_size.append([H, W])
    cropped_size.append([h, w])

    assert coord[0] + h < H and coord[1] + w < W

    print(coord, [H, W], [h, w])

data = pd.DataFrame()
data['name'] = fnames
data['position'] = positions
data['original_size'] = original_size
data['cropped_size'] = cropped_size
data.to_csv('subimage_position.csv')
