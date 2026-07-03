import csv
import itertools
import math
import os
from collections import Counter

import cv2
import numpy as np


DATASET_FILE = "knn_color_dataset.csv"
LABELS = ["F", "U", "B", "D", "R", "L"]

face_coordinates = np.array([
    [(50, 50), (110, 110)], [(210, 50), (270, 110)], [(370, 50), (430, 110)],
    [(50, 210), (110, 270)], [(210, 210), (270, 270)], [(370, 210), (430, 270)],
    [(50, 370), (110, 430)], [(210, 370), (270, 430)], [(370, 370), (430, 430)],
])


def rgb_to_hsv_feature(rgb):
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

    # Use cos/sin for hue so red near 0 and red near 360 stay close.
    return np.array([r, g, b, math.cos(radians), math.sin(radians), s, v], dtype=float)


def extract_patch_features(image, coordinate):
    (x1, y1), (x2, y2) = coordinate
    patch = image[y1:y2, x1:x2]
    rgb_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    mean_rgb = cv2.mean(rgb_patch)[:3]
    return rgb_to_hsv_feature(mean_rgb)


def extract_patch_feature_set(image, coordinate):
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
        features.append(rgb_to_hsv_feature(mean_rgb))
    return np.array(features, dtype=float)


def load_dataset(dataset_file=DATASET_FILE):
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


def normalize_features(train_features, sample_features):
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std[std == 0] = 1.0
    return (train_features - mean) / std, (sample_features - mean) / std


def knn_label_scores(sample_features, train_features, train_labels, k=5):
    normalized_train, normalized_samples = normalize_features(train_features, sample_features)
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


def choose_unique_center_mapping(center_scores):
    best_perm = None
    best_cost = float("inf")
    for physical_labels in itertools.permutations(LABELS):
        cost = 0.0
        for face_index, physical_label in enumerate(physical_labels):
            cost += center_scores[face_index][physical_label]
        if cost < best_cost:
            best_cost = cost
            best_perm = physical_labels

    # Physical color -> current cube face label.
    return {
        physical_label: face_label
        for physical_label, face_label in zip(best_perm, LABELS)
    }


def balance_to_nine(label_scores):
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


def color_classification(k=5, dataset_file=DATASET_FILE):
    train_features, train_labels = load_dataset(dataset_file)
    physical_scores = []

    for face_index in range(1, 7):
        image = cv2.imread("cube_test%d.jpg" % face_index)
        if image is None:
            raise FileNotFoundError("cube_test%d.jpg not found" % face_index)

        for coordinate in face_coordinates:
            sample_features = extract_patch_feature_set(image, coordinate)
            physical_scores.append(
                knn_label_scores(sample_features, train_features, train_labels, k)
            )

    center_indices = [4, 13, 22, 31, 40, 49]
    center_scores = [physical_scores[index] for index in center_indices]
    physical_to_face = choose_unique_center_mapping(center_scores)

    face_scores = []
    for scores in physical_scores:
        remapped = {}
        for physical_label, score in scores.items():
            remapped[physical_to_face[physical_label]] = score
        face_scores.append(remapped)

    color_pred = balance_to_nine(face_scores)

    counts = Counter(color_pred)
    for label in LABELS:
        if counts[label] != 9:
            print("KNN warning: %s count is %d, expected 9" % (label, counts[label]))

    return np.array(color_pred, dtype=str)
