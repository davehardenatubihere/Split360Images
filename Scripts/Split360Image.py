import os
import numpy as np
import cv2
from scipy.interpolate import RectBivariateSpline
from math import pi, acos


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
    # Ensure the save folder exists
    os.makedirs(save_folder, exist_ok=True)

    frames = []
    height, width, _ = panorama.shape
    d = distance
    xp_max, yp_max = (1 + d) / d, (1 + d) / d
    xp_domain = xp_max * (np.arange(-1., 1., 2. / height) + 1.0 / height)
    yp_domain = yp_max * (np.arange(-1., 1., 2. / height) + 1.0 / height)
    delta_rad = 2 * pi / width

    for face in range(4):
        print(f'Generating stereo image {face}')
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

        # Save the split image to the specified folder
        filename = os.path.join(save_folder, f'split_{face}_{d}.jpg')
        cv2.imwrite(filename, output_img)
        print(f"Saved: {filename}")

        frames.append(output_img)
        panorama = np.concatenate((panorama[:, int(width / 4):, :], panorama[:, :int(width / 4), :]), axis=1)

    return frames


if __name__ == "__main__":
    # Set the path to your panorama image here
    image_path = "../Full 360 Images/city.png"  # Change this to the actual path of your image
    save_folder = "split_images/city"  # Folder to save the split images

    # Load the image
    pano_image = cv2.imread(image_path)

    if pano_image is not None:
        print("Processing panorama...")
        panotostereo(pano_image, save_folder=save_folder, distance=2)
        print("Image split completed. Saved in:", save_folder)
    else:
        print(f"Error: Could not load image from {image_path}. Check the file path.")
