import cv2
import os
import numpy as np
import pandas as pd
import re
import skimage.io
import skimage.segmentation


def preprocessing_images(image_name, parent_folder, destination_folder):
    # Read the image
    image = cv2.imread(f"{parent_folder}/{image_name}")

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Image smoothing
    therhold_image = min_pixel_value(gray_image)
    blurred_image = cv2.GaussianBlur(therhold_image, (15, 15), 0)

    # Find the center (where the laser points at)
    center = find_center(blurred_image)

    # Crop the image and then resize it around the center
    cropped_image = crop_center(blurred_image, (256, 256), center=center)

    # Save the image
    cv2.imwrite(f'{destination_folder}/{image_name}', cropped_image)


def min_pixel_value(image):
    min_value = 30

    image = np.where(image < min_value, 0, image)

    return image


def max_pixel_value(image):
    max_value = 253

    image = np.where(image > max_value, max_value, image)

    return image


def find_center(image):
    # Threshold the image
    image_clear = skimage.segmentation.clear_border(image >= 254.0)

    # Find the coordinates of the thresholded image
    y, x = np.nonzero(image_clear)

    # Check if x and y are empty
    if len(x) == 0 or len(y) == 0:
        return (0, 0)  # Default value or any other appropriate value

    # Find average
    xmean = int(np.mean(x))
    ymean = int(np.mean(y))

    return (xmean, ymean)


def crop_center(image, new_dimensions, center=None):
    # Get dimensions of the image
    height, width = image.shape[:2]

    # Get desired dimensions
    crop_width = new_dimensions[0]
    crop_height = new_dimensions[1]
    
    # Calculate the center of the image
    if center == None:
        center_x, center_y = width // 2, height // 2
    else:
        center_x, center_y = center[0], center[1]
    
    # Calculate the top-left corner of the crop area
    x_start = max(center_x - crop_width // 2, 0)
    y_start = max(center_y - crop_height // 2, 0)
    
    # Ensure the cropped area does not go outside the image
    x_end = min(x_start + crop_width, width)
    y_end = min(y_start + crop_height, height)
    
    # Crop the image
    cropped_image = image[y_start:y_end, x_start:x_end]
    
    return cropped_image


def imagefolder2csv(folder):
    all_files = os.listdir(folder)
    image_array = []
    names = []
    # pattern = r"perfil_(.*?)_2024"

    for file in all_files:
        if file.endswith(('.png')) and "background" not in file and "laser" not in file:
            image = cv2.imread(f"{folder}/{file}", cv2.IMREAD_GRAYSCALE)
            image_array.append(image.flatten())

            # match = re.search(pattern, file)
            # if match:
            #     substring = match.group(1)
            #     names.append(substring)
            # else:
            #     print("Pattern not found in the filename.")
            
    image_array = np.array(image_array)
    # names_array = np.array(names)
    # total_array = np.c_[image_array, names_array]
    df = pd.DataFrame(image_array)
    df.to_csv(f"{folder}/{folder}.csv", sep=";", header=False, index=False)


def delete_img_background(image, parent_folder):
    all_files = os.listdir(parent_folder)

    for file in all_files:
        if "background" in file:
            background = cv2.imread(f"{parent_folder}/{file}")

    final_image = image - background

    return final_image

parent_folder = "original_images"
destination_folder = "preprocessed_images"
all_files = os.listdir(parent_folder)

for file in all_files:
    if file.endswith("png"):
        preprocessing_images(file, parent_folder, destination_folder)
    else:
        print(f"Failed to read image {file}")

imagefolder2csv(destination_folder)
imagefolder2csv("buenas")