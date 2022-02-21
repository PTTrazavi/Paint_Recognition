import os
import numpy as np
import tensorflow as tf
from model import customized_Lightweight_model
from modules.utils import set_memory_growth


def Get_Model(image_size = 112):
    # build the model
    model = customized_Lightweight_model(
        input_shape = (image_size, image_size, 3),
    )
    # get model weight
    ckpt_path = tf.train.latest_checkpoint(model_ckpt_path)
    model.load_weights(ckpt_path).expect_partial()
    return model

def main():
    model = Get_Model()
    model.summary()

    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS,   # enable TensorFlow ops.
    ]
    converter.allow_custom_ops=True
    converter.experimental_new_converter =True
    tflite_model = converter.convert()
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    set_memory_growth()
    model_ckpt_path = './checkpoints/00/'
    tflite_path = "./checkpoints/model_00.tflite"
    main()
