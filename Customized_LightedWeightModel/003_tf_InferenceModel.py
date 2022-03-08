import os
import cv2
import numpy as np
import tensorflow as tf
from model import customized_Lightweight_model
from modules import dataset
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, precision_score
from tqdm import tqdm
import matplotlib.pyplot as plt


def Get_Model(image_size = 112):
    # build model
    model = customized_Lightweight_model(
        input_shape = (image_size, image_size, 3),
    )
    # get model weight
    ckpt_path = tf.train.latest_checkpoint(model_ckpt_path)
    model.load_weights(ckpt_path).expect_partial()
    return model

def TestingModel():
    def Predict_Dataset(model):
        def Model_Inference(model, resized_Images):
            # inference by model
            Pred_Probability = model(resized_Images)
            # get label
            classes = np.argmax(Pred_Probability, 1)
            return classes

        Pred_Classes = np.zeros([val_samples, ])
        GT_Classes = np.zeros([val_samples, ])

        test_dataset, test_samples = dataset.load_dataset(pkl_files=pkl_files, batch_size=batch_size, mode='val')
        test_dataset = iter(test_dataset)
        for idx in tqdm(range(0, val_samples, batch_size)):
            inputs, labels = next(test_dataset)
            Pred = Model_Inference(model, inputs)
            Pred_Classes[idx:idx + batch_size] = Pred
            GT_Classes[idx:idx + batch_size] = labels

        return GT_Classes, Pred_Classes

    model = Get_Model()
    GT_Classes, Pred_Classes = Predict_Dataset(model)
    print(GT_Classes.shape)
    print(Pred_Classes.shape)
    CF = confusion_matrix(GT_Classes, Pred_Classes)
    Accuracy = accuracy_score(GT_Classes, Pred_Classes)
    F1_score = f1_score(GT_Classes, Pred_Classes, average='macro')
    Recall = recall_score(GT_Classes, Pred_Classes, average='macro')
    Precision = precision_score(GT_Classes, Pred_Classes, average='macro')

    print('-'*100)
    print('CF: ')
    print(CF)
    print('F1_score: ', F1_score)
    print('Accuracy: ', Accuracy)
    print('Recall: ', Recall)
    print('Precision: ', Precision)
    print('='*100)

def TestingImage(ImagePath):
    def Model_Inference(model, ImagePath):
        def Image_preprocessing(ImagePath, ImageSize=112):
            Images = tf.image.decode_jpeg(tf.io.read_file(ImagePath), channels=3)
            resized_Images = tf.image.resize(Images, (ImageSize, ImageSize))
            # resized_Images = resized_Images / 255.
            resized_Images = (resized_Images - 127.5) / 127.5
            # expand 1 dimension(batch dimension)
            resized_Images = tf.expand_dims(resized_Images, axis=0)
            return resized_Images

        # get input image
        resized_Image = Image_preprocessing(ImagePath)
        # inference by model
        Pred_Probability = model(resized_Image)
        print(Pred_Probability)
        # get label
        classes = np.argmax(Pred_Probability, 1)
        return classes

    def show_image(ImagePath, Pred_classes):
        fig = plt.figure(figsize=(6, 6))
        Images = cv2.imread(ImagePath)
        Images = cv2.cvtColor(Images, cv2.COLOR_BGR2RGB)
        resized_Images = cv2.resize(Images, (96, 96))
        plt.xticks([])
        plt.yticks([])
        plt.imshow(resized_Images)
        plt.suptitle(f'Pred_Classes: {Pred_classes}',fontsize=32)
        plt.show()
        fig.savefig('inference_result.jpg')

    model = Get_Model()
    classes = Model_Inference(model, ImagePath)
    show_image(ImagePath, Pred_classes=classes[0])

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # Use GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # set parameters
    pkl_files = './data/dataset_00.pkl'
    model_ckpt_path = './checkpoints/00/'
    val_samples = 114
    batch_size = 2
    # Run through all validation dataset
    TestingModel()
    # Test 1 image
    TestingImage('./20220107JURASSIC_PIC_rename/01/01/square-uvc-sample-52974328703895612.jpg')
