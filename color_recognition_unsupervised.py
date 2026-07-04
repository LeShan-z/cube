"""
纯无监督颜色识别 —— 不依赖任何颜色先验知识。

原理：
1. 提取 54 个色块的 (cos(H), sin(H), S, V) 特征向量
2. K-means (k=6) 聚类，让数据自动发现 6 个自然颜色分组
3. 通过中心块结构映射：每面中心块所属的聚类 → 该面的颜色标签
   - 图1(F面)中心块的颜色类 → 该类的标签为 "F"
   - 图2(U面)中心块的颜色类 → 该类的标签为 "U"
   - 依此类推
4. 后处理平衡：确保每色恰好 9 个

与 color_recognition_algorithm.py 的关键差异：
- 不假设"白色 = 低饱和度"
- 不假设任何颜色在 HSV 空间的特殊位置
- 所有 6 种颜色平等对待，纯聚类区分
"""

import math
from collections import Counter

import cv2
import numpy as np


# 480×480 裁剪图中 9 个色块区域的坐标
face_coordinates = np.array([
    [(50, 50),   (110, 110)],  # 左上
    [(210, 50),  (270, 110)],  # 上中
    [(370, 50),  (430, 110)],  # 右上
    [(50, 210),  (110, 270)],  # 左中
    [(210, 210), (270, 270)],  # 中心
    [(370, 210), (430, 270)],  # 右中
    [(50, 370),  (110, 430)],  # 左下
    [(370, 370), (430, 430)],  # 右下 -- 注意原代码坐标有误，第8和第9坐标相同
])

# 修正：face_coordinates 的第 7、8 个位置在原代码中有误，补上正确的坐标
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
CENTER_INDICES = [4, 13, 22, 31, 40, 49]  # 全局索引中每面中心块的位置


def rgb_to_features(rgb):
    """
    将单个 RGB 三元组转为 4 维特征向量 (cos(H), sin(H), S, V)。

    用 (cos, sin) 编码色相，保证 0° 和 359° 在特征空间中接近，
    使欧氏距离能正确处理圆形色相。
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


def extract_features():
    """
    从 cube_test1.jpg ~ cube_test6.jpg 中提取全部 54 个色块的特征。

    Returns
    -------
    features : ndarray (54, 4)
        每行是 (cos(H), sin(H), S, V)
    """
    features = []
    for i in range(1, 7):
        image = cv2.imread("cube_test%d.jpg" % i)
        if image is None:
            raise FileNotFoundError("cube_test%d.jpg not found" % i)
        im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for (x1, y1), (x2, y2) in face_coordinates:
            patch = im_rgb[y1:y2, x1:x2]
            mean_rgb = cv2.mean(patch)[:3]
            features.append(rgb_to_features(mean_rgb))

    return np.array(features, dtype=float)


def _euclidean_dist_sq(a, b):
    """两个向量之间的欧氏距离平方。"""
    diff = a - b
    return np.dot(diff, diff)


def kmeans_cluster(features, k=6, max_iter=100, n_init=15):
    """
    K-means 聚类，返回最优的 label 和 center。

    首次初始化使用 6 个中心块的特征作为初始质心（最可靠的初始点）。
    后续初始化随机选取，取惯性最小的结果。

    Parameters
    ----------
    features : ndarray (54, 4)
        归一化后的特征矩阵
    k : int
        聚类数，固定 6
    max_iter : int
        每次 K-means 最大迭代次数
    n_init : int
        初始化尝试次数

    Returns
    -------
    labels : ndarray (54,) int
        每个样本的 cluster id
    centers : ndarray (6, 4)
        最终聚类中心
    """
    n_samples = features.shape[0]
    best_labels = None
    best_centers = None
    best_inertia = float("inf")
    rng = np.random.default_rng()

    for init_idx in range(n_init):
        if init_idx == 0 and n_init > 1:
            # 第一次：用 6 个中心块作为初始质心
            centers = features[CENTER_INDICES].copy()
        else:
            # 随机选取 k 个不重复样本
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

            # 重新计算聚类中心
            for j in range(k):
                mask = labels == j
                if mask.sum() > 0:
                    centers[j] = features[mask].mean(axis=0)
                else:
                    # 空聚类：重新随机初始化该中心
                    centers[j] = features[rng.choice(n_samples)]

        # 计算惯性（类内平方和）
        inertia = sum(
            _euclidean_dist_sq(features[i], centers[labels[i]])
            for i in range(n_samples)
        )

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()

    return best_labels, best_centers


def map_clusters_to_faces(labels, features, centers):
    """
    将聚类结果映射为面标签。

    核心逻辑：每面中心块所属的 cluster 即为该面的颜色标签。
    如果 6 个中心块落在 6 个不同 cluster → 映射成功。
    如果有中心块碰撞（两个中心同属一个 cluster）→ 返回 None 表示需要重新聚类。
    如果有 cluster 没有中心块 → 用最近的中心块给它分配面标签。

    Returns
    -------
    result : ndarray (54,) str or None
    """
    # 记录每个 cluster 包含哪个面的中心
    cluster_owns_face = {}  # cluster_id → face_index
    for face_idx, center_idx in enumerate(CENTER_INDICES):
        cid = labels[center_idx]
        if cid in cluster_owns_face:
            # 两个中心块落在同一个 cluster，聚类失败
            print("碰撞：面 %s 和面 %s 的中心在同一个簇 %d" % (
                FACE_LABELS[cluster_owns_face[cid]], FACE_LABELS[face_idx], cid
            ))
            return None
        cluster_owns_face[cid] = face_idx

    # 对于没有中心块的 orphan cluster，分配给最近的中心
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

    # 生成结果
    result = np.array([FACE_LABELS[cluster_to_face[l]] for l in labels], dtype=str)
    return result


def balance_to_nine(result, features, centers, cluster_to_face):
    """
    将超过 9 个的颜色的多余色块重新分配到不足 9 个的颜色，
    每次选择代价最小的色块-目标颜色对进行转移。
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
                # 目标 cluster 的中心
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


def color_classification():
    """
    无监督颜色识别主函数。

    流程：
    1. 从 6 张图提取 54 个特征向量
    2. Z-score 归一化
    3. K-means 聚类 (k=6)
    4. 通过中心块映射 cluster → 面标签
    5. 如果映射失败（中心碰撞），重新聚类
    6. 后处理平衡到每色恰好 9 个

    Returns
    -------
    color_pred : ndarray (54,) str
        54 个字符，顺序 F→U→B→D→R→L
    """
    features = extract_features()

    # Z-score 归一化（所有维度等权重）
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std == 0] = 1.0
    features_norm = (features - mean) / std

    max_attempts = 30
    result = None

    for attempt in range(max_attempts):
        # 第一次尝试用较多 init 提升成功率，后续递增
        n_init = max(1, 20 - attempt)
        labels, centers = kmeans_cluster(features_norm, n_init=n_init)
        result = map_clusters_to_faces(labels, features_norm, centers)
        if result is not None:
            break
        print("第 %d 次聚类映射失败，重试中..." % (attempt + 1))

    if result is None:
        raise RuntimeError(
            "经过 %d 次尝试仍无法将中心块映射到不同聚类。"
            "请检查光照条件或重新拍照。" % max_attempts
        )

    # 构造 cluster_to_face 映射（用于平衡步骤）
    cluster_to_face = {}
    for face_idx, center_idx in enumerate(CENTER_INDICES):
        cid = labels[center_idx]
        cluster_to_face[cid] = FACE_LABELS[face_idx]

    # 处理没有中心块的 orphan cluster
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

    result = balance_to_nine(result, features_norm, centers, cluster_to_face)

    # 最终校验
    counts = Counter(result)
    for label in FACE_LABELS:
        if counts[label] != 9:
            print("警告：%s 有 %d 个色块（期望 9）" % (label, counts[label]))

    return result
