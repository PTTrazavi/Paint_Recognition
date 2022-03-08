import os
import random
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import cv2


def create_annotation_file(root_folder, target, contain_50, save_path, VALIDATION_RATIO = 0.15):
    # check target images
    image_path = []
    labels = []
    corners = {"01":"up-left","02":"up-right","03":"down-left","04":"down-right"}

    for k, v in corners.items():
        files = [os.path.join(root_folder, target, k, i) for i in os.listdir(os.path.join(root_folder, target, k)) if not i.startswith('.')]
        print(f"{len(files)} images in {v} folder")
        image_path = image_path + files
        labels = labels + [int(k[-1])]*len(files)

    positive = len(labels)

    # get all paints folder
    folders = [os.path.join(root_folder, i) for i in os.listdir(os.path.join(root_folder)) if not i.startswith('.')]
    print(f"dataset contains {len(folders)} paints")

    if contain_50:
        files = [os.path.join(root_folder, target, "50", i) for i in os.listdir(os.path.join(root_folder, target, "50")) if not i.startswith('.')]
        # print(f"{len(files)} images in 50 folder")
        image_path = image_path + files
        labels = labels + [0]*len(files)
        negative_left = positive - len(files)
        # calculate how many images should be sampled in each corner folder
        sample_count = round(negative_left / ((len(folders)-1)*4))
        for f in folders:
            if os.path.join(root_folder, target) not in f:
                for k, v in corners.items():
                    files = [os.path.join(f, k, i) for i in os.listdir(os.path.join(f, k)) if not i.startswith('.')]
                    files = random.choices(files, k=sample_count)
                    # print(f"{len(files)} images in {v} folder")
                    image_path = image_path + files
                    labels = labels + [0]*len(files)
    else:
        # calculate how many images should be sampled in each corner folder
        sample_count = round(positive / ((len(folders)-1)*4))
        for f in folders:
            if os.path.join(root_folder, target) not in f:
                for k, v in corners.items():
                    files = [os.path.join(f, k, i) for i in os.listdir(os.path.join(f, k)) if not i.startswith('.')]
                    files = random.choices(files, k=sample_count)
                    # print(f"{len(files)} images in {v} folder")
                    image_path = image_path + files
                    labels = labels + [0]*len(files)

    print(f"positive sample: {positive} images.")
  	# print(f"image path example:{image_path[:3]}")
    print(f"negative sample: {len(labels)-positive} images.")
  	# print(f"image path example:{image_path[-3:]}")

    X_train, X_val, y_train, y_val = train_test_split(image_path, labels, test_size=VALIDATION_RATIO)
    assert(len(X_train) == len(y_train))
    assert(len(X_val) == len(y_val))
    print(f"train size: {len(y_train)}")
    print(f"val size: {len(y_val)}" )
    print(f"image path examples: {X_train[:3]}")
    print(f"label examples: {y_train[:10]}")

    data_file = {
        'Training_Set':{'path':X_train, 'label':np.array(y_train)},
        'Validation_Set':{'path':X_val, 'label':np.array(y_val)}
        }
    pickle.dump(data_file, open(save_path, 'wb'))
    print(f"saved image paths and labels in {save_path}")

    # calculate image mean and std
    pixels = 0
    std_sum = 0
    count = 0
    for i in X_train:
        image = cv2.imread(i)
        pixels = pixels + np.sum(image)
        count = count + image.size
    mean = pixels/count
    print(f"training data mean: {round(mean,1)}")
    for i in X_train:
        image = cv2.imread(i)
        std_sum = std_sum + np.sum((image - mean)**2)
    std = (std_sum/count)**0.5
    print(f"training data std: {round(std,1)}")


if __name__== '__main__':
    root_folder = "../Customized_LightedWeightModel/20220107JURASSIC_PIC_rename"
    target = "00"
    contain_50 = True
    save_path = 'data/dataset_00.pkl'
    create_annotation_file(root_folder, target, contain_50, save_path)
