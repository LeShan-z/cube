"""
统一颜色识别模块 —— 整合三种颜色识别算法。

使用方式：直接调用对应的函数名即可切换算法。
  - color_classification_hsv()      带先验知识无监督版（白色饱和度+色相聚类）
  - color_classification_knn()      有监督KNN版（需CSV训练数据集）
  - color_classification_unsupervised()  纯无监督K-means版

所有函数接口一致：无参数，返回 shape=(54,) 的字符串数组，顺序 F→U→B→D→R→L。
"""

import csv
import itertools
import math
import os
from collections import Counter

import cv2
import numpy as np


# ============================================================
# 共享常量 —— 三个版本共用
# ============================================================

# 480×480 裁剪图中9个色块区域的坐标（左上角, 右下角）
face_coordinates = np.array([
    [(50, 50),   (110, 110)],
    [(210, 50),  (270, 110)],
    [(370, 50),  (430, 110)],
    [(50, 210),  (110, 270)],
    [(210, 210), (270, 270)],
    [(370, 210), (430, 270)],
    [(50, 370),  (110, 430)],
    [(210, 370), (270, 430)],
    [(370, 370), (430, 430)],
])

FACE_LABELS = ["F", "U", "B", "D", "R", "L"]
CENTER_INDICES = [4, 13, 22, 31, 40, 49]  # 6个面中心块在54个色块中的全局索引


# ============================================================
# 版本一：带先验知识无监督版（白色饱和度 + 色相聚类）
# 原始文件：color_recognition_algorithm.py
# ============================================================

def _maxmin3(a, b, c):
    """三数求最大最小值，用于 RGB → HSV 转换。"""
    if a < b:
        max_value, min_value = b, a
    else:
        max_value, min_value = a, b
    if max_value < c:
        max_value = c
    if min_value > c:
        min_value = c
    return max_value, min_value


def _RGB2HSV(RGB_feature):
    """批量 RGB → HSV 转换。"""
    x, y = RGB_feature.shape
    HSV_feature = np.zeros((x, y))

    for i in range(x):
        r = RGB_feature[i, 0] / 255
        g = RGB_feature[i, 1] / 255
        b = RGB_feature[i, 2] / 255
        max_value, min_value = _maxmin3(r, g, b)
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


def _cluster_hsv(HSV_Feature):
    """
    白色饱和度分离 + 5色色相圆形聚类 + 固定面顺序标签映射。
    核心假设：白色饱和度最低；其余5色通过色相H做圆形聚类区分。
    """
    index = np.zeros(54, dtype=int)
    for i in range(54):
        index[i] = i

    # --- 步骤1：饱和度排序，分离白色 ---
    # 饱和度最低的9个 → 白色
    for i in range(9):
        for j in range(54 - i - 1):
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

    # --- 步骤2：用非白色中心块的色相作为5个聚类中心的初始值 ---
    h_center = np.zeros(5)
    h_center_old = np.array([-1, 0, 0, 0, 0], dtype=float)
    group = np.zeros((5, 45), dtype=int)
    group_count = np.zeros(5, dtype=int)
    h_center_index = 0
    for i in range(6):
        color_index = 9 * i + 4
        if color_index != white_center_index:
            h_center[h_center_index] = HSV_Feature[color_index][0]
            h_center_index += 1

    # --- 步骤3：迭代聚类（最多50次）---
    MAX_LOOP = 50
    for loop in range(MAX_LOOP):
        for j in range(5):
            group_count[j] = 0
        for i in range(45):
            h = HSV_Feature[index[i]][0]
            min_dist = float("inf")
            min_dist_group = 0
            for j in range(5):
                h0 = h_center[j]
                diff = abs(h - h0) % 360
                dist = 360 - diff if diff > 180 else diff
                if dist < min_dist:
                    min_dist = dist
                    min_dist_group = j
            group[min_dist_group][group_count[min_dist_group]] = index[i]
            group_count[min_dist_group] += 1

        # 用圆形均值重新计算每个聚类的色相中心
        for j in range(5):
            sum_cos = 0
            sum_sin = 0
            pi = math.pi
            for i in range(group_count[j]):
                hsv_color_index = group[j][i]
                a = HSV_Feature[hsv_color_index][0] * (pi / 180)
                sum_cos += math.cos(a)
                sum_sin += math.sin(a)
            if group_count[j] > 0:
                a = math.atan2(sum_sin, sum_cos) * (180 / pi)
                h_center[j] = (a + 360) % 360

        change = 0
        for j in range(5):
            if h_center_old[j] != h_center[j]:
                change += 1

        if change == 0:
            print("聚类中心已经没有改变, 循环： " + str(loop))
            break
        else:
            for j in range(5):
                h_center_old[j] = h_center[j]

    # --- 步骤4：按固定面顺序分配标签 ---
    color_str = np.array(["F", "U", "B", "D", "R", "L"])
    color_pred = np.empty(54, dtype=str)
    group_count_index = 0
    for i in range(6):
        center_index = 9 * i + 4
        if center_index != white_center_index:
            if group_count[group_count_index] != 9:
                print("分类错误(某类色块个数有误), 调整灯光, 重新拍照")

            center_in_group = 0
            for j in range(group_count[group_count_index]):
                color_index = group[group_count_index][j]
                color_pred[color_index] = color_str[i]
                if color_index == center_index:
                    center_in_group += 1
            if center_in_group != 1:
                print("分类错误(某面中心块错分类), 调整灯光, 重新拍照")

            group_count_index += 1

    # 空白位置（白色块）用白色中心块所在面标签填充
    for i in range(54):
        if color_pred[i] == "":
            color_pred[i] = color_str[int((white_center_index - 4) / 9)]

    return color_pred


def color_classification_hsv():
    """
    带先验知识无监督版（白色饱和度 + 色相聚类）。

    原理：白色饱和度最低，先分离白色；剩余5色用色相(H)做圆形聚类；
    按固定拍摄面顺序(F→U→B→D→R→L)分配标签。

    Returns
    -------
    color_pred : ndarray (54,) str
    """
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
    HSV_feature = _RGB2HSV(RGB_feature)
    color_pred = _cluster_hsv(HSV_feature)

    return color_pred


# ============================================================
# 版本二：有监督KNN版（CSV训练数据集 + 中心块排列匹配）
# 原始文件：color_recognition_knn.py
# ============================================================

DATASET_FILE = "knn_color_dataset.csv"
LABELS = ["F", "U", "B", "D", "R", "L"]


def _rgb_to_hsv_feature_knn(rgb):
    """RGB → 7维特征向量 (R, G, B, cos(H), sin(H), S, V)。"""
    r, g, b = [float(v) / 255.0 for v in rgb]
    max_value = max(r, g, b)
    min_value = min(r, g, b)
    delta = max_value - min_value

    if delta == 0:
        h = 0.0
    elif max_value == r:
        h = ((g - b) / delta) * 60.0
    elif max_value == g:
        h = 120.0 + ((b - r) / delta) * 60.0
    else:
        h = 240.0 + ((r - g) / delta) * 60.0
    if h < 0:
        h += 360.0

    s = 0.0 if max_value == 0 else delta / max_value
    v = max_value
    radians = math.radians(h)

    return np.array([r, g, b, math.cos(radians), math.sin(radians), s, v], dtype=float)


def _extract_patch_features_knn(image, coordinate):
    """提取单个色块的7维特征（整体均值）。"""
    (x1, y1), (x2, y2) = coordinate
    patch = image[y1:y2, x1:x2]
    rgb_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    mean_rgb = cv2.mean(rgb_patch)[:3]
    return _rgb_to_hsv_feature_knn(mean_rgb)


def _extract_patch_feature_set_knn(image, coordinate):
    """
    提取单个色块的多尺度特征集（10个特征向量）。
    1个整体均值 + 3×3九宫格子区域均值，增强抗干扰能力。
    """
    (x1, y1), (x2, y2) = coordinate
    patch = image[y1:y2, x1:x2]
    height, width, _ = patch.shape
    rgb_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

    regions = [(0, 0, width, height)]
    for row in range(3):
        for col in range(3):
            sx1 = int(width * col / 3)
            sx2 = int(width * (col + 1) / 3)
            sy1 = int(height * row / 3)
            sy2 = int(height * (row + 1) / 3)
            regions.append((sx1, sy1, sx2, sy2))

    features = []
    for sx1, sy1, sx2, sy2 in regions:
        sample_patch = rgb_patch[sy1:sy2, sx1:sx2]
        mean_rgb = cv2.mean(sample_patch)[:3]
        features.append(_rgb_to_hsv_feature_knn(mean_rgb))
    return np.array(features, dtype=float)


def _load_dataset_knn(dataset_file=DATASET_FILE):
    """加载KNN训练数据集。"""
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(
            "%s not found. Run collect_knn_dataset.py first." % dataset_file
        )

    features = []
    labels = []
    with open(dataset_file, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(row["label"])
            features.append([
                float(row["r"]),
                float(row["g"]),
                float(row["b"]),
                float(row["h_cos"]),
                float(row["h_sin"]),
                float(row["s"]),
                float(row["v"]),
            ])

    if not features:
        raise ValueError("%s is empty." % dataset_file)

    return np.array(features, dtype=float), np.array(labels)


def _normalize_features_knn(train_features, sample_features):
    """Z-score归一化。"""
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std[std == 0] = 1.0
    return (train_features - mean) / std, (sample_features - mean) / std


def _knn_label_scores(sample_features, train_features, train_labels, k=5):
    """计算单个色块对各标签的KNN距离得分。"""
    normalized_train, normalized_samples = _normalize_features_knn(train_features, sample_features)
    scores = {}
    for label in LABELS:
        class_features = normalized_train[train_labels == label]
        class_distances = []
        for sample in normalized_samples:
            distances = np.linalg.norm(class_features - sample, axis=1)
            nearest = np.sort(distances)[:k]
            class_distances.append(float(np.mean(nearest)))
        scores[label] = float(np.mean(class_distances))
    return scores


def _choose_unique_center_mapping(center_scores):
    """
    通过6个中心块的6!全排列搜索，找到物理颜色标签→标准面标签的最优一一映射。
    """
    best_perm = None
    best_cost = float("inf")
    for physical_labels in itertools.permutations(LABELS):
        cost = 0.0
        for face_index, physical_label in enumerate(physical_labels):
            cost += center_scores[face_index][physical_label]
        if cost < best_cost:
            best_cost = cost
            best_perm = physical_labels

    return {
        physical_label: face_label
        for physical_label, face_label in zip(best_perm, LABELS)
    }


def _balance_to_nine_knn(label_scores):
    """
    贪心算法确保每色恰好9个色块。
    每次选择代价增量最小的色块从超量标签迁移到不足标签。
    """
    labels = [min(scores, key=scores.get) for scores in label_scores]
    counts = Counter(labels)

    while any(counts[label] > 9 for label in LABELS):
        over_labels = [label for label in LABELS if counts[label] > 9]
        under_labels = [label for label in LABELS if counts[label] < 9]
        if not over_labels or not under_labels:
            break

        best_move = None
        best_delta = float("inf")
        for index, current_label in enumerate(labels):
            if current_label not in over_labels:
                continue
            for target_label in under_labels:
                delta = label_scores[index][target_label] - label_scores[index][current_label]
                if delta < best_delta:
                    best_delta = delta
                    best_move = (index, current_label, target_label)

        if best_move is None:
            break

        index, old_label, new_label = best_move
        labels[index] = new_label
        counts[old_label] -= 1
        counts[new_label] += 1

    return labels


def color_classification_knn(k=5, dataset_file=DATASET_FILE):
    """
    有监督KNN版（CSV训练数据集 + 中心块排列匹配）。

    需要预先运行 collect_knn_dataset.py 生成 knn_color_dataset.csv。

    Parameters
    ----------
    k : int
        KNN近邻数，默认5
    dataset_file : str
        训练数据集路径

    Returns
    -------
    color_pred : ndarray (54,) str
    """
    train_features, train_labels = _load_dataset_knn(dataset_file)
    physical_scores = []

    for face_index in range(1, 7):
        image = cv2.imread("cube_test%d.jpg" % face_index)
        if image is None:
            raise FileNotFoundError("cube_test%d.jpg not found" % face_index)

        for coordinate in face_coordinates:
            sample_features = _extract_patch_feature_set_knn(image, coordinate)
            physical_scores.append(
                _knn_label_scores(sample_features, train_features, train_labels, k)
            )

    center_scores = [physical_scores[index] for index in CENTER_INDICES]
    physical_to_face = _choose_unique_center_mapping(center_scores)

    face_scores = []
    for scores in physical_scores:
        remapped = {}
        for physical_label, score in scores.items():
            remapped[physical_to_face[physical_label]] = score
        face_scores.append(remapped)

    color_pred = _balance_to_nine_knn(face_scores)

    counts = Counter(color_pred)
    for label in LABELS:
        if counts[label] != 9:
            print("KNN warning: %s count is %d, expected 9" % (label, counts[label]))

    return np.array(color_pred, dtype=str)


# ============================================================
# 版本三：纯无监督K-means版（无任何颜色先验知识）
# 原始文件：color_recognition_unsupervised.py
# ============================================================

def _rgb_to_features_unsupervised(rgb):
    """
    将单个RGB三元组转为4维特征向量 (cos(H), sin(H), S, V)。
    用(cos, sin)编码色相解决0°/359°的圆形边界问题。
    """
    r, g, b = [float(v) / 255.0 for v in rgb]
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    delta = max_val - min_val

    if delta == 0:
        h = 0.0
    elif max_val == r:
        h = ((g - b) / delta) * 60.0
    elif max_val == g:
        h = 120.0 + ((b - r) / delta) * 60.0
    else:
        h = 240.0 + ((r - g) / delta) * 60.0
    if h < 0:
        h += 360.0

    s = 0.0 if max_val == 0 else delta / max_val
    v = max_val

    rad = math.radians(h)
    return np.array([math.cos(rad), math.sin(rad), s, v], dtype=float)


def _extract_features_unsupervised():
    """从 cube_test1.jpg ~ cube_test6.jpg 提取全部54个色块的4维特征。"""
    features = []
    for i in range(1, 7):
        image = cv2.imread("cube_test%d.jpg" % i)
        if image is None:
            raise FileNotFoundError("cube_test%d.jpg not found" % i)
        im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for (x1, y1), (x2, y2) in face_coordinates:
            patch = im_rgb[y1:y2, x1:x2]
            mean_rgb = cv2.mean(patch)[:3]
            features.append(_rgb_to_features_unsupervised(mean_rgb))

    return np.array(features, dtype=float)


def _euclidean_dist_sq(a, b):
    """两个向量的欧氏距离平方（省去开方运算）。"""
    diff = a - b
    return np.dot(diff, diff)


def _kmeans_cluster(features, k=6, max_iter=100, n_init=15):
    """
    K-means聚类，返回最优的labels和centers。
    首次初始化用6个中心块特征作为初始质心，后续随机初始化，取惯性最小结果。
    """
    n_samples = features.shape[0]
    best_labels = None
    best_centers = None
    best_inertia = float("inf")
    rng = np.random.default_rng()

    for init_idx in range(n_init):
        if init_idx == 0 and n_init > 1:
            centers = features[CENTER_INDICES].copy()
        else:
            idx = rng.choice(n_samples, k, replace=False)
            centers = features[idx].copy()

        labels = np.zeros(n_samples, dtype=int)

        for iteration in range(max_iter):
            changed = False
            for i in range(n_samples):
                min_dist = float("inf")
                best_label = 0
                for j in range(k):
                    d = _euclidean_dist_sq(features[i], centers[j])
                    if d < min_dist:
                        min_dist = d
                        best_label = j
                if labels[i] != best_label:
                    labels[i] = best_label
                    changed = True

            if not changed:
                break

            for j in range(k):
                mask = labels == j
                if mask.sum() > 0:
                    centers[j] = features[mask].mean(axis=0)
                else:
                    centers[j] = features[rng.choice(n_samples)]

        inertia = sum(
            _euclidean_dist_sq(features[i], centers[labels[i]])
            for i in range(n_samples)
        )

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()

    return best_labels, best_centers


def _map_clusters_to_faces(labels, features, centers):
    """
    将聚类结果映射为面标签。
    核心：每面中心块所属的cluster → 该面的颜色标签。
    如果两个中心块碰撞（同属一个cluster）→ 返回None需要重试。
    """
    cluster_owns_face = {}
    for face_idx, center_idx in enumerate(CENTER_INDICES):
        cid = labels[center_idx]
        if cid in cluster_owns_face:
            print("碰撞：面 %s 和面 %s 的中心在同一个簇 %d" % (
                FACE_LABELS[cluster_owns_face[cid]], FACE_LABELS[face_idx], cid
            ))
            return None
        cluster_owns_face[cid] = face_idx

    cluster_to_face = {}
    for cid in range(6):
        if cid in cluster_owns_face:
            cluster_to_face[cid] = cluster_owns_face[cid]
        else:
            best_face = 0
            best_dist = float("inf")
            for face_idx in range(6):
                center_feat = features[CENTER_INDICES[face_idx]]
                d = _euclidean_dist_sq(centers[cid], center_feat)
                if d < best_dist:
                    best_dist = d
                    best_face = face_idx
            cluster_to_face[cid] = best_face

    result = np.array([FACE_LABELS[cluster_to_face[l]] for l in labels], dtype=str)
    return result


def _balance_to_nine_unsupervised(result, features, centers, cluster_to_face):
    """
    将超过9个的颜色色块重新分配到不足9个的颜色。
    每次选择代价（到目标聚类中心距离）最小的色块进行迁移。
    """
    counts = Counter(result)
    over_labels = [l for l in FACE_LABELS if counts[l] > 9]
    under_labels = [l for l in FACE_LABELS if counts[l] < 9]

    while over_labels and under_labels:
        best_sticker = None
        best_target = None
        best_cost = float("inf")

        for i, current_label in enumerate(result):
            if current_label not in over_labels:
                continue
            for target_label in under_labels:
                target_cid = [c for c, f in cluster_to_face.items() if f == target_label][0]
                cost = _euclidean_dist_sq(features[i], centers[target_cid])
                if cost < best_cost:
                    best_cost = cost
                    best_sticker = i
                    best_target = target_label

        if best_sticker is None:
            break

        old_label = result[best_sticker]
        result[best_sticker] = best_target
        counts[old_label] -= 1
        counts[best_target] += 1

        over_labels = [l for l in FACE_LABELS if counts[l] > 9]
        under_labels = [l for l in FACE_LABELS if counts[l] < 9]

    return result


def color_classification_unsupervised():
    """
    纯无监督K-means版（无任何颜色先验知识）。

    原理：
    1. 提取54个色块的(cosH, sinH, S, V)特征
    2. Z-score归一化后K-means(k=6)聚类
    3. 通过中心块结构自动映射cluster→面标签
    4. 贪心平衡确保每色恰好9个

    不假设白色=低饱和度，不假设任何颜色在HSV空间的位置。

    Returns
    -------
    color_pred : ndarray (54,) str
    """
    features = _extract_features_unsupervised()

    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std == 0] = 1.0
    features_norm = (features - mean) / std

    max_attempts = 30
    result = None
    labels = None
    centers = None

    for attempt in range(max_attempts):
        n_init = max(1, 20 - attempt)
        labels, centers = _kmeans_cluster(features_norm, n_init=n_init)
        result = _map_clusters_to_faces(labels, features_norm, centers)
        if result is not None:
            break
        print("第 %d 次聚类映射失败，重试中..." % (attempt + 1))

    if result is None:
        raise RuntimeError(
            "经过 %d 次尝试仍无法将中心块映射到不同聚类。"
            "请检查光照条件或重新拍照。" % max_attempts
        )

    cluster_to_face = {}
    for face_idx, center_idx in enumerate(CENTER_INDICES):
        cid = labels[center_idx]
        cluster_to_face[cid] = FACE_LABELS[face_idx]

    for cid in range(6):
        if cid not in cluster_to_face:
            best_face = 0
            best_dist = float("inf")
            for face_idx in range(6):
                center_feat = features_norm[CENTER_INDICES[face_idx]]
                d = _euclidean_dist_sq(centers[cid], center_feat)
                if d < best_dist:
                    best_dist = d
                    best_face = face_idx
            cluster_to_face[cid] = FACE_LABELS[best_face]

    result = _balance_to_nine_unsupervised(result, features_norm, centers, cluster_to_face)

    counts = Counter(result)
    for label in FACE_LABELS:
        if counts[label] != 9:
            print("警告：%s 有 %d 个色块（期望 9）" % (label, counts[label]))

    return result
