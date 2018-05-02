import glob
import os
import pandas as pd
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

range_x=20
range_y=10

def get_index(angle):
    return int((angle+1)/0.02)


def load_dataset(camera_angle,lap, np_counter_array, aug_trans = True,aug_bright = True, aug_flip = True, aug_add = False):

    LIMIT = 200
    LARGE_INDEX = 101
    SMALL_INDEX = 0

    if lap == "LEFT":
        csv_path = 'Datasets/driving_log_LEFT.csv'
    elif lap == "RIGHT":
        csv_path = 'Datasets/driving_log_RIGHT.csv'
    elif lap == "mond":
        csv_path = 'Datasets/driving_log_mond.csv'
    elif lap == "mond2":
        csv_path = 'Datasets/driving_log_mond2.csv'
    elif lap == "mond3":
        csv_path = 'Datasets/driving_log_mond3.csv'
    elif lap == "mond4":
        csv_path = 'Datasets/driving_log_mond4.csv'
    elif lap == "test":
        csv_path = 'Datasets/driving_log_test.csv'
    else:
        print("No dataset loaded")


    data_files = pd.read_csv(csv_path, index_col = False)
    data_files.columns = ['center','left','right','steer','throttle','break', 'speed']

    data_size = len(data_files)

    np_images = np.zeros((1, 64, 64, 3))
    np_steering = np.zeros(1)

    skip_count = 0

    ########### CODE TO PREVENT EMPTY IMAGE IN np_images[0] ##############

    image = cv2.imread(data_files[camera_angle][0].strip())
    # Transform from BGR to RGB
    image = image[...,::-1]
    steer = data_files['steer'][0]

    counter = 0

    while image is None:
        counter += 1
        image = cv2.imread(data_files[camera_angle][counter].strip())
        image = image[...,::-1]
        steer = data_files['steer'][counter]

    # To ensure no images with angles larger than abs(1) are added to np_images
    if camera_angle == "left" and steer > 0.8:
        skip_count += 1
    elif camera_angle == "right" and steer < -0.8:
        skip_count += 1

    if camera_angle == "left":
        steer += 0.15
    elif camera_angle == "right":
        steer -= 0.15

    # Change brightness
    if aug_bright:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
        hsv[:,:,2] =  hsv[:,:,2] * ratio
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # Resize to (64, 64, 3)
    shape = image.shape
    image = image[int(math.floor(shape[0]/4)):shape[0]-25, 0:shape[1]]
    image = cv2.resize(image,(64,64), interpolation=cv2.INTER_AREA)
    image = image/255.-.5

    image = np.array(image)

    # Intermediate step to allow concatenation of image to np_images
    temp_img_array = np.zeros((1,64,64,3))
    temp_img_array[0] = image

    np_images[0] = image
    np_steering[0] = steer

    index = get_index(steer)
    np_counter_array[index] += 1

    #######################################################################


    for i_elem in range(counter, data_size):

        image = cv2.imread(data_files[camera_angle][i_elem].strip())
        image = image[...,::-1]

        if image is not None:

            steer = data_files['steer'][i_elem]

            # To ensure no images with angles larger than abs(1) are added to np_images
            if camera_angle == "left" and steer > 0.8:
                skip_count += 1
                continue
            elif camera_angle == "right" and steer < -0.8:
                skip_count += 1
                continue

            if camera_angle == "left":
                steer += 0.15
            elif camera_angle == "right":
                steer -= 0.15

            index = get_index(steer)

            if np_counter_array[index] < LIMIT:

                if aug_bright:
                    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
                    hsv[:,:,2] =  hsv[:,:,2] * ratio
                    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

                shape = image.shape
                image = image[int(math.floor(shape[0]/4)):shape[0]-25, 0:shape[1]]
                image = cv2.resize(image,(64,64), interpolation=cv2.INTER_AREA)
                image = image/255.-.5

                if aug_flip:
                    if np.random.rand() < 0.25:
                        image = cv2.flip(image, 1)
                        steer = -steer

                if aug_trans:
                    trans_x = range_x * (np.random.rand() - 0.5)
                    trans_y = range_y * (np.random.rand() - 0.5)
                    steer += trans_x * 0.002
                    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
                    height, width = image.shape[:2]
                    trans_x = range_x * (np.random.rand() - 0.5)
                    trans_y = range_y * (np.random.rand() - 0.5)
                    steer += trans_x * 0.002
                    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
                    height, width = image.shape[:2]
                    image = cv2.warpAffine(image, trans_m, (width, height))

                image = np.array(image)

                # Intermediate step to allow concatenation of image to np_images
                temp_img_array[0] = image
                np_images = np.concatenate((np_images, temp_img_array))
                np_steering = np.append(np_steering, steer)
                np_counter_array[index] += 1

                if aug_add:
                    if abs(steer) > 0.25 and np_counter_array[index] < LIMIT:
                        larger_index = get_index(steer+0.02)
                        if larger_index < LARGE_INDEX:
                            if np_counter_array[larger_index] < LIMIT and ((steer+0.02) < 2.0):
                                np_images = np.concatenate((np_images, temp_img_array))
                                np_steering = np.append(np_steering, steer+0.02)
                                np_counter_array[larger_index] += 1

                        smaller_index = get_index(steer-0.02)
                        if smaller_index >= SMALL_INDEX:
                            if np_counter_array[smaller_index] < LIMIT and ((steer-0.02) > 0.0):
                                np_images = np.concatenate((np_images, temp_img_array))
                                np_steering = np.append(np_steering, steer-0.02)
                                np_counter_array[smaller_index] += 1


        else:
            skip_count += 1

    print("-----SKIPPED ", skip_count, " ITEMS")
    print("------- Number of images: ", len(np_images), " --- Number of angles: ", len(np_steering))

    return np_images, np_steering, np_counter_array;
