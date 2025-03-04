import os
import numpy as np
import cv2
from scipy.interpolate import RectBivariateSpline
from math import pi, acos
import torch
from ultralytics import YOLO


# Function to calculate projection angle
def projection_angle(x, d):
    x_max = (1 + d) / d
    numerator = -2 * d * x ** 2 + 2 * (d + 1) * np.sqrt((1 - d ** 2) * x ** 2 + (d + 1) ** 2)
    denominator = 2 * (x ** 2 + (d + 1) ** 2)
    if 0 < x < x_max:
        return acos(numerator / denominator)
    elif x < 0:
        return -acos(numerator / denominator)
    elif x == x_max:
        return pi / 2
    else:
        raise Exception('Invalid input arguments')


# Function to split the panoramic image and save images in a specified folder
def panotostereo(panorama, save_folder="split_images", distance=2):
    os.makedirs(save_folder, exist_ok=True)  # Ensure the save folder exists

    frames = []
    height, width, _ = panorama.shape
    d = distance
    xp_max, yp_max = (1 + d) / d, (1 + d) / d
    xp_domain = xp_max * (np.arange(-1., 1., 2. / height) + 1.0 / height)
    yp_domain = yp_max * (np.arange(-1., 1., 2. / height) + 1.0 / height)
    delta_rad = 2 * pi / width

    for face in range(4):
        print(f'Generating split image {face}')
        output_img = np.zeros((height, height, 3))

        interpolate_0 = RectBivariateSpline(np.arange(height), np.arange(width), panorama[:, :, 0])
        interpolate_1 = RectBivariateSpline(np.arange(height), np.arange(width), panorama[:, :, 1])
        interpolate_2 = RectBivariateSpline(np.arange(height), np.arange(width), panorama[:, :, 2])

        pano_x = np.zeros((height, 1))
        pano_y = np.zeros((height, 1))

        for j, xp in enumerate(xp_domain):
            pano_x[j] = (width / 2.0 + (projection_angle(xp, d) / delta_rad))

        for i, yp in enumerate(yp_domain):
            pano_y[i] = height / 2.0 + (projection_angle(yp, d) / delta_rad)

        output_img[:, :, 0] = interpolate_0(pano_y, pano_x)
        output_img[:, :, 1] = interpolate_1(pano_y, pano_x)
        output_img[:, :, 2] = interpolate_2(pano_y, pano_x)

        filename = os.path.join(save_folder, f'split_{face}.jpg')
        cv2.imwrite(filename, output_img)
        print(f"Saved: {filename}")

        frames.append(filename)
        panorama = np.concatenate((panorama[:, int(width / 4):, :], panorama[:, :int(width / 4), :]), axis=1)

    return frames


# Function to run YOLO object detection on split images
def detect_objects_on_splits(split_images, model_path="../YOLO/yolov8n.pt"):
    model = YOLO(model_path)

    for img_path in split_images:
        image = cv2.imread(img_path)

        if image is None:
            print(f"Error: Could not load split image {img_path}. Skipping.")
            continue

        results = model(image)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                label = f"{model.names[class_id]}: {confidence:.2f}"

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        output_path = img_path.replace(".jpg", "_detected.jpg")
        cv2.imwrite(output_path, image)
        print(f"Saved detected image: {output_path}")

        cv2.imshow(f"Detection {img_path}", image)
        cv2.waitKey(500)  # Show image briefly
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Set the path to the 360-degree image
    image_path = "../Full 360 Images/bend5.png"
    save_folder = "split_images/street"

    # Load the 360-degree image
    pano_image = cv2.imread(image_path)

    if pano_image is not None:
        print("Processing panorama...")
        split_images = panotostereo(pano_image, save_folder=save_folder, distance=2)
        print("Image split completed. Running object detection...")
        detect_objects_on_splits(split_images)
        print("Object detection completed.")
    else:
        print(f"Error: Could not load image from {image_path}. Check the file path.")
