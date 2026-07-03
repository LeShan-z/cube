import math

import cv2
import numpy as np


# Coordinates of the 9 stickers in the cropped 480x480 cube face image.
face_coordinates = np.array([
    [(50, 50), (110, 110)], [(210, 50), (270, 110)], [(370, 50), (430, 110)],
    [(50, 210), (110, 270)], [(210, 210), (270, 270)], [(370, 210), (430, 270)],
    [(50, 370), (110, 430)], [(210, 370), (270, 430)], [(370, 370), (430, 430)],
])


def maxmin3(a, b, c):
    if a < b:
        max_value = b
        min_value = a
    else:
        max_value = a
        min_value = b
    if max_value < c:
        max_value = c
    if min_value > c:
        min_value = c
    return max_value, min_value


def RGB2HSV(RGB_feature):
    x, y = RGB_feature.shape
    HSV_feature = np.zeros((x, y))

    for i in range(0, x):
        r = RGB_feature[i, 0] / 255
        g = RGB_feature[i, 1] / 255
        b = RGB_feature[i, 2] / 255
        max_value, min_value = maxmin3(r, g, b)
        delta = max_value - min_value

        if delta == 0:
            HSV_feature[i, 0] = 0
        else:
            if r == max_value:
                HSV_feature[i, 0] = ((g - b) / delta) * 60
            elif g == max_value:
                HSV_feature[i, 0] = 120 + (((b - r) / delta) * 60)
            elif b == max_value:
                HSV_feature[i, 0] = 240 + (((r - g) / delta) * 60)
            if HSV_feature[i, 0] < 0:
                HSV_feature[i, 0] += 360
        if max_value == 0:
            HSV_feature[i, 1] = 0
        else:
            HSV_feature[i, 1] = delta / max_value

        HSV_feature[i, 2] = max_value

    return HSV_feature


def cluster(HSV_Feature):
    index = np.zeros(54, dtype=int)
    for i in range(0, 54):
        index[i] = i

    # White stickers have the lowest saturation. Put the 9 lowest-S stickers at
    # index[45] to index[53].
    for i in range(0, 9):
        for j in range(0, 54 - i - 1):
            if HSV_Feature[index[j]][1] < HSV_Feature[index[j + 1]][1]:
                tmp = index[j]
                index[j] = index[j + 1]
                index[j + 1] = tmp

    white_center_index = -1
    for i in range(45, 54):
        if index[i] % 9 == 4:
            white_center_index = index[i]
            print("白色中心块序号为： " + str(white_center_index))

    if white_center_index == -1:
        print("未找到白色中心块，请重新拍照")

    # Use the non-white center stickers as the initial hue cluster centers.
    h_center = np.zeros(5)
    h_center_old = np.array([-1, 0, 0, 0, 0], dtype=float)
    group = np.zeros((5, 45), dtype=int)
    group_count = np.zeros(5, dtype=int)
    h_center_index = 0
    for i in range(0, 6):
        color_index = 9 * i + 4
        if color_index != white_center_index:
            h_center[h_center_index] = HSV_Feature[color_index][0]
            h_center_index += 1

    MAX_LOOP = 50
    for loop in range(0, MAX_LOOP):
        for j in range(0, 5):
            group_count[j] = 0
        for i in range(0, 45):
            h = HSV_Feature[index[i]][0]
            min_dist = float("inf")
            min_dist_group = 0
            for j in range(0, 5):
                h0 = h_center[j]
                diff = abs(h - h0) % 360
                if diff > 180:
                    dist = 360 - diff
                else:
                    dist = diff
                if dist < min_dist:
                    min_dist = dist
                    min_dist_group = j
            group[min_dist_group][group_count[min_dist_group]] = index[i]
            group_count[min_dist_group] += 1

        # Recompute each hue center with a circular mean.
        for j in range(0, 5):
            sum_cos = 0
            sum_sin = 0
            pi = math.pi
            for i in range(0, group_count[j]):
                hsv_color_index = group[j][i]
                a = HSV_Feature[hsv_color_index][0] * (pi / 180)
                sum_cos += math.cos(a)
                sum_sin += math.sin(a)

            if group_count[j] > 0:
                a = math.atan2(sum_sin, sum_cos) * (180 / pi)
                h_center[j] = (a + 360) % 360

        change = 0
        for j in range(0, 5):
            if h_center_old[j] != h_center[j]:
                change += 1

        if change == 0:
            print("聚类中心已经没有改变, 循环： " + str(loop))
            break
        else:
            for j in range(0, 5):
                h_center_old[j] = h_center[j]

    # Face order: F U B D R L.
    color_str = np.array(["F", "U", "B", "D", "R", "L"])
    color_pred = np.empty(54, dtype=str)
    group_count_index = 0
    for i in range(0, 6):
        center_index = 9 * i + 4
        if center_index != white_center_index:
            if group_count[group_count_index] != 9:
                print("分类错误(某类色块个数有误), 调整灯光, 重新拍照")

            center_in_group = 0
            for j in range(0, group_count[group_count_index]):
                color_index = group[group_count_index][j]
                color_pred[color_index] = color_str[i]
                if color_index == center_index:
                    center_in_group += 1
            if center_in_group != 1:
                print("分类错误(某面中心块错分类), 调整灯光, 重新拍照")

            group_count_index += 1

    for i in range(0, 54):
        if color_pred[i] == "":
            color_pred[i] = color_str[int((white_center_index - 4) / 9)]

    return color_pred


def color_classification():
    RGB_feature_temp = []
    for i in range(1, 7):
        image = cv2.imread("cube_test%d.jpg" % i)
        im_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for cordinates in face_coordinates:
            face_RGB = im_RGB[
                cordinates[0, 1]:cordinates[1, 1],
                cordinates[0, 0]:cordinates[1, 0],
            ]

            temp_rgb = cv2.mean(face_RGB)
            temp_rgb = temp_rgb[0:3]
            RGB_feature_temp.append(temp_rgb)

    RGB_feature = np.array(RGB_feature_temp)
    HSV_feature = RGB2HSV(RGB_feature)
    color_pred = cluster(HSV_feature)

    return color_pred
