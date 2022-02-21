import os
import cv2
import numpy as np
from tqdm import tqdm
from modules.dataset import load_dataset


def main(annot_file):
    if not os.path.isdir('./Previewed_image'):
        os.mkdir('./Previewed_image')

    train_dataset, train_samples = load_dataset(annot_file, mode='train', batch_size=16)

    for idx, parsed_record in tqdm(enumerate(train_dataset)):
        Img, label = parsed_record
        recon_img = np.array(Img[0].numpy() * 255, 'uint8')
        cv2.imwrite(f'./Previewed_image/{idx:04d}.jpg', cv2.cvtColor(recon_img, cv2.COLOR_RGB2BGR))
        if idx == 16:
            break

if __name__ == '__main__':
    annot_file = './data/dataset_00.pkl'
    main(annot_file)
