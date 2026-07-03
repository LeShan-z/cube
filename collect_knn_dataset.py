import csv
import os
import time

import cv2

from color_recognition_knn import LABELS, face_coordinates, rgb_to_hsv_feature


DATASET_FILE = "knn_color_dataset.csv"
IMAGE_PATTERN = "cube_test%d.jpg"


def crop_cube_frame(frame):
    height, width, _ = frame.shape
    x, y = int(width / 2), int(height / 2)
    return frame[y - 240:y + 240, x - 240:x + 240]


def wait_frames(cam, count):
    frame = None
    for _ in range(count):
        ok, frame = cam.read()
        if not ok:
            raise RuntimeError("Failed to read camera frame.")
    return frame


def capture_solved_cube_images():
    import serial

    myserial = serial.Serial("/dev/ttyUSB0", 115200, timeout=0.5)
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise RuntimeError("Failed to open the camera.")

    try:
        ok, frame = cam.read()
        if not ok:
            raise RuntimeError("Failed to read camera frame.")

        commands_after_capture = [
            b"#BU!",
            b"#BUM!",
            b"#MBU!",
            b"#IBU!",
            b"#BBUM!",
            None,
        ]

        for face_index, command in enumerate(commands_after_capture, start=1):
            frame = wait_frames(cam, 10)
            cube_frame = crop_cube_frame(frame)
            filename = IMAGE_PATTERN % face_index
            cv2.imwrite(filename, cube_frame)
            print("saved %s" % filename)

            if command is not None:
                myserial.write(command)
                time.sleep(0.5)
                wait_frames(cam, 80)
    finally:
        cam.release()


def iter_patch_samples(image, coordinate):
    (x1, y1), (x2, y2) = coordinate
    patch = image[y1:y2, x1:x2]
    height, width, _ = patch.shape

    # Whole sticker region plus a 3x3 grid inside it. This gives the KNN model
    # several brightness/shadow variations for each sticker.
    regions = [(0, 0, width, height)]
    for row in range(3):
        for col in range(3):
            sx1 = int(width * col / 3)
            sx2 = int(width * (col + 1) / 3)
            sy1 = int(height * row / 3)
            sy2 = int(height * (row + 1) / 3)
            regions.append((sx1, sy1, sx2, sy2))

    rgb_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    for sx1, sy1, sx2, sy2 in regions:
        sample_patch = rgb_patch[sy1:sy2, sx1:sx2]
        mean_rgb = cv2.mean(sample_patch)[:3]
        yield rgb_to_hsv_feature(mean_rgb)


def build_dataset_from_images(dataset_file=DATASET_FILE):
    rows = []
    for face_index, label in enumerate(LABELS, start=1):
        filename = IMAGE_PATTERN % face_index
        image = cv2.imread(filename)
        if image is None:
            raise FileNotFoundError("%s not found." % filename)

        for sticker_index, coordinate in enumerate(face_coordinates):
            for sample_index, feature in enumerate(iter_patch_samples(image, coordinate)):
                rows.append({
                    "label": label,
                    "face_index": face_index,
                    "sticker_index": sticker_index,
                    "sample_index": sample_index,
                    "r": feature[0],
                    "g": feature[1],
                    "b": feature[2],
                    "h_cos": feature[3],
                    "h_sin": feature[4],
                    "s": feature[5],
                    "v": feature[6],
                })

    with open(dataset_file, "w", newline="") as f:
        fieldnames = [
            "label", "face_index", "sticker_index", "sample_index",
            "r", "g", "b", "h_cos", "h_sin", "s", "v",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("wrote %d samples to %s" % (len(rows), dataset_file))


def main():
    capture = os.environ.get("CAPTURE", "1") != "0"
    if capture:
        capture_solved_cube_images()
    build_dataset_from_images()


if __name__ == "__main__":
    main()
