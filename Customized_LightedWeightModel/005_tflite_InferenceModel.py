import numpy as np
import tensorflow as tf

def inference(interpreter, input_details, output_details, ImagePath):
	def Image_preprocessing(ImagePath, ImageSize=112):
		Images = tf.image.decode_jpeg(tf.io.read_file(ImagePath), channels=3)
		resized_Images = tf.image.resize(Images, (ImageSize, ImageSize))
		# resized_Images = resized_Images / 255.
		resized_Images = (resized_Images - 127.5) / 127.5
		# expand 1 dimension(batch dimension)
		resized_Images = tf.expand_dims(resized_Images, axis=0)
		return resized_Images
	# get preprocessed input image
	resized_Image_ndarray = np.array(Image_preprocessing(ImagePath))
	# set input image
	interpreter.set_tensor(input_details[0]['index'], resized_Image_ndarray)
	interpreter.invoke()
	Pred_Probability = interpreter.get_tensor(output_details[0]['index'])
	# Pred_Probability  = np.squeeze(output_data)
	# get label
	classes = np.argmax(Pred_Probability)

	print('='*100)
	print(f'Pred_Probability: \n{np.round(Pred_Probability, decimals=4)}')
	print('-'*100)
	print(f'classes: \n{classes}')
	print('='*100)

def main(model_path, ImagePath):
	# import model (tf 2.8.0)
	interpreter = tf.lite.Interpreter(model_path)
	interpreter.allocate_tensors()
	# get input output tensor
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()
	# inference
	inference(interpreter, input_details, output_details, ImagePath)

if __name__ == '__main__':
	ImagePath = './20220107JURASSIC_PIC_rename/00/03/uvc-sample-2019914016662063000.jpg'
	model_path = "./checkpoints/model_00.tflite"
	main(model_path, ImagePath)
