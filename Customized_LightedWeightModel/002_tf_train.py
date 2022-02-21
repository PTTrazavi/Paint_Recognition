import os
import warnings
warnings.filterwarnings("ignore")

from modules.dataset import load_dataset
from modules.utils import set_memory_growth
from model import customized_Lightweight_model

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.experimental import CosineDecayRestarts
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler, ReduceLROnPlateau

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # choose GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    set_memory_growth()

    model = customized_Lightweight_model(
        input_shape = (image_size, image_size, 3),
    )

    model.summary()
    # get training data
    train_dataset, train_samples = load_dataset(pkl_files=pkl_files, batch_size=batch_size, mode='train')
    val_dataset, val_samples = load_dataset(pkl_files=pkl_files, batch_size=val_batch_size, mode='val')

    steps_per_epoch = (train_samples // batch_size)+1 if train_samples % batch_size != 0 else (train_samples // batch_size)
    validation_steps = val_samples // val_batch_size

    # set learning rate schedule
    ## CosineDecayRestarts
    # Scheduler = CosineDecayRestarts(
    #     initial_learning_rate=base_lr,
    #     first_decay_steps=20
    # )
    # reduce_lr =  LearningRateScheduler(Scheduler)
    ## ReduceLROnPlateau
    reduce_lr = ReduceLROnPlateau(patience=5, factor=0.1, verbose=1), #
    # set checkpoint
    mc_callback = ModelCheckpoint(
        model_ckpt,
        monitor='val_loss',
        verbose=1,
        save_weights_only=True,
        save_best_only=True
    )
    # set tensorboard
    tb_callback = TensorBoard(
        log_dir='logs/',
        update_freq=batch_size * 5,
        profile_batch=0
    )
    # set optimizer, loss, metrics
    model.compile(
        optimizer=Adam(learning_rate=base_lr),
        loss=SparseCategoricalCrossentropy(),
        metrics=['accuracy'],
    )
    # train the model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=[mc_callback, tb_callback, reduce_lr],
    )

    print("[*] training done!")


if __name__ == '__main__':
    batch_size = 32
    val_batch_size = 2
    image_size = 112
    epochs = 60
    base_lr = 2e-4
    pkl_files = './data/dataset_00.pkl'
    model_ckpt = './checkpoints/00/best.ckpt'
    main()
