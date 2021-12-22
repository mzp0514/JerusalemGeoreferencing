import numpy as np
import cv2
import pandas as pd
import glob
import os


def render(img, points):
    res = np.copy(img)
    for i in range(points.shape[0]):
        x = min(max(0, points[i][0]), img.shape[0] - 1)
        y = min(max(0, points[i][1]), img.shape[1] - 1)
        res[x][y] = 255
    return res


def read_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = 255 * (img > 100)
    img = img.astype('uint8')
    return img


def getCentroidByContour(contour):
    M = cv2.moments(contour)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return (cx, cy)


def getCentroidByPoints(points):
    (rows, cols) = points.T
    return [np.mean(rows), np.mean(cols)]


def select_keypoints(contours):
    (rows, cols) = contours.T
    key_points = []

    # down = contours[np.argmax(rows)]
    # right = contours[np.argmax(cols)]
    # up = contours[np.argmin(rows)]
    # left = contours[np.argmin(cols)]

    left_down = contours[np.argmin(cols - 2 * rows)]
    right_down = contours[np.argmax(0.5 * rows + cols)]
    right_up = contours[np.argmax(cols - 2 * rows)]
    left_up = contours[np.argmin(0.5 * rows + cols)]

    cent = getCentroidByPoints(contours)

    key_points.append(cent)
    key_points.append(left_up)
    key_points.append(left_down)
    key_points.append(right_down)
    key_points.append(right_up)

    return np.array(key_points)


def overlappingArea(contours, contours2, shape):
    contours_cp = np.copy(contours)
    contours2_cp = np.copy(contours2)
    res = np.zeros((shape[0] * 2, shape[1] * 2), dtype=np.uint8)
    res2 = np.zeros((shape[0] * 2, shape[1] * 2), dtype=np.uint8)

    contours_cp += np.array([shape[0] // 2, shape[1] // 2])
    contours2_cp += np.array([shape[0] // 2, shape[1] // 2])
    res = render(res, contours_cp.astype(np.int32))
    res2 = render(res2, contours2_cp.astype(np.int32))

    kernel = np.ones((3, 3))
    res = cv2.dilate(res, kernel, iterations=1)
    contours_, _ = cv2.findContours(res, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)

    contours2_, _ = cv2.findContours(res2, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_NONE)

    assert len(contours_) == 1
    assert len(contours2_) == 1

    cv2.floodFill(res, None, getCentroidByContour(contours_[0]), 255)
    cv2.floodFill(res2, None, getCentroidByContour(contours2_[0]), 255)

    overlapping = np.sum(np.logical_and(res, res2))

    return overlapping, (overlapping / cv2.countNonZero(res2) +
                         overlapping / cv2.countNonZero(res)) / 2


def findScaling(contours, contours2):
    anchors = select_keypoints(contours)
    anchors2 = select_keypoints(contours2)

    dx = anchors[3][0] - anchors[4][0]
    dy = anchors[3][1] - anchors[1][1]
    dx2 = anchors2[3][0] - anchors2[4][0]
    dy2 = anchors2[3][1] - anchors2[1][1]

    scale_y = dy2 / dy
    scale_x = dx2 / dx
    return scale_x, scale_y


def findMaxOverlapping(contours, contours2, shape, vertices, num_it=4):

    for it in range(num_it):

        scale_x, scale_y = findScaling(contours, contours2)
        # scale
        (rows, cols) = contours.T
        contours = np.array([rows * scale_x, cols * scale_y]).T

        (row_vert, col_vert) = vertices.T
        vertices = np.array([row_vert * scale_x, col_vert * scale_y]).T

        best_M = None
        best_overlapping = 0
        best_translation = None

        contours_copy = np.copy(contours)

        ones = np.ones(shape=(contours.shape[0], 1))
        anchors = select_keypoints(contours)
        anchors2 = select_keypoints(contours2)

        for i, anchor in enumerate(anchors):

            # translation
            anchor2 = anchors2[i]
            translation = anchor2 - anchor
            contours = contours_copy + translation

            # rotation
            for k in range(-6, 7):
                angle = k * (num_it - it)
                M = cv2.getRotationMatrix2D(tuple(anchor2), angle, 1)
                dst = M.dot(np.hstack([contours, ones]).T).T

                area, _ = overlappingArea(dst, contours2, shape)
                if area > best_overlapping:
                    best_overlapping = area
                    best_M = M
                    best_translation = translation

        contours = best_M.dot(
            np.hstack([contours_copy + best_translation, ones]).T).T

        ones = np.ones(shape=(vertices.shape[0], 1))
        vertices = best_M.dot(
            np.hstack([vertices + best_translation, ones]).T).T

    _, score = overlappingArea(contours, contours2, shape)

    return contours.astype(np.int32), score, vertices


in_path = './boundaries'
out_path = './boundaries_aligned'

in_files = glob.glob(in_path + '/*')
coordinates = []
fnames = []
scores = []

in_files = sorted(in_files)
for f in in_files:
    if f.split('.')[-1] in ['jpg', 'png']:

        print(f)
        img_name = f.split(os.path.sep)[-1]

        img = read_img(f)
        img2 = read_img('anchor.png')

        original_img = cv2.imread('./original.png')

        contours = np.transpose(np.nonzero(img))
        contours2 = np.transpose(np.nonzero(img2))

        vertices = np.array([[img.shape[0] - 1, img.shape[1] - 1],
                             [0, img.shape[1] - 1], [img.shape[0] - 1, 0],
                             [0, 0]])

        contours, score, vertices = findMaxOverlapping(contours, contours2,
                                                       img.shape, vertices)

        (rows, cols) = vertices.T
        rows = img2.shape[0] - 1 - rows

        # 35.222 long , 31.787 lat
        A = [35.222, 31.769]
        B = [35.24, 31.787]
        dx = abs(A[0] - B[0])
        assert img2.shape[0] == img2.shape[1]
        assert abs(A[0] - B[0]) == abs(A[1] - B[1])
        scale = dx / img2.shape[0]

        rows = rows * scale + A[1]
        cols = cols * scale + A[0]

        vertices = np.array([cols, rows]).T

        scores.append(score)
        fnames.append(f)
        coordinates.append(vertices.tolist())

        origin_img_cp = original_img.copy()
        for i in range(contours.shape[0]):
            if contours[i][0] < 0 or contours[i][0] >= img2.shape[
                    0] or contours[i][1] < 0 or contours[i][1] >= img2.shape[1]:
                continue

            img2[contours[i][0]][contours[i][1]] = 150
            original_img[contours[i][0]][contours[i][1]] = [0, 0, 255]
            original_img[contours[i][0] + 1][contours[i][1]] = [0, 0, 255]
            original_img[contours[i][0]][contours[i][1] + 1] = [0, 0, 255]
            original_img[contours[i][0] + 1][contours[i][1] + 1] = [0, 0, 255]

        cv2.imwrite(out_path + '/' + img_name, img2)
        cv2.imwrite('images_aligned/' + img_name, original_img)

data = pd.DataFrame()
data['name'] = fnames
data['score'] = scores
data['coordinate'] = coordinates
data.to_csv('coordinates.csv')
