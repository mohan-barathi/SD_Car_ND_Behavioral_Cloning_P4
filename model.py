# Behavioral Cloning
# Udacity Self Driving Car Nano Degree
# Term 1 ; Project 4

# Author : Mohan Barathi

# Import all the required modules
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, Cropping2D, Dropout, Flatten, Dense, Lambda, MaxPooling2D
from keras import optimizers
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from sklearn.utils import shuffle


# Global variables:
global data_path

def fetch_data():
	"""
	Read the csv file for image path and steering angles
	"""
	global data_path
	samples = []
	with open(data_path+'driving_log.csv') as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			samples.append(line)
	samples = samples[1:]          # Removing the header line

	# make the image paths in csv as relative paths
	for sample in samples:
		for i in range(0,3):
			sample[i] = data_path+sample[i].strip()

	return samples


def create_nvidia_model(input_shape):
	"""
	This function creates a cnn model, as described in Nvidia's paper
	http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
	Many Dropouts are added for regularization.
	This function returns a compiled model, that can be used for fit ot fit_generator
	"""
	nvidia_model = Sequential()

	nvidia_model.add(Lambda(lambda x: (x-128)/128, input_shape=input_shape))

	nvidia_model.add(Cropping2D(((70, 25), (0, 0))))

	nvidia_model.add(Conv2D(24, (5, 5), activation="relu", strides=(2, 2)))
	#nvidia_model.add(Dropout(0.2))

	nvidia_model.add(Conv2D(36, (5, 5), activation="relu", strides=(2, 2)))
	#nvidia_model.add(Dropout(0.2))

	nvidia_model.add(Conv2D(48, (5, 5), activation="relu", strides=(2, 2)))
	nvidia_model.add(Dropout(0.1))

	nvidia_model.add(Conv2D(64, (3, 3), activation="relu"))
	nvidia_model.add(Dropout(0.1))

	nvidia_model.add(Conv2D(64, (3, 3), activation="relu"))
	nvidia_model.add(Dropout(0.1))

	nvidia_model.add(Flatten())

	nvidia_model.add(Dense(100, activation='relu'))
	nvidia_model.add(Dropout(0.3))

	nvidia_model.add(Dense(50, activation='relu'))
	nvidia_model.add(Dropout(0.3))

	nvidia_model.add(Dense(10, activation='relu'))
	nvidia_model.add(Dense(1))
	opt = optimizers.adam(lr=0.002)

	nvidia_model.compile(optimizer=opt, loss='mse')

	return nvidia_model
	
	
	
def generator(samples, batch_size=32, generator_name = "Generator", center_only=False, augment_images=True):
	"""
	This is a factory function, that returns generator functions
	configured based on the input arguments and flags.

	Param-in : 
		samples       : The data samples which will be used by the generator functions
		batch_size    : The batch size based on which the data is loaded into memory
		center_only   : The flag to set, if only center images should be considered
		augment_images: The flag to set, if the augment images should be produced
		
	Param-out :
		generator_function	:	The function that yields a small sub-set of data, 
								from a huge set of data, to avoid memory consumption.

	"""

	def generator_function():
		"""
		Generator function, that yields a samll batch of data, whenever invoked.
		This generator also handles the left / right camera images, assigning an
		arbitrary steering angle value for those images, and augmentation of images 
		using a simple left-to-right flip.
		"""
		num_samples = len(samples)
		global data_path
		
		# Loop forever so the generator never terminates
		while 1: 
			shuffle(samples)
			processed_samples = 0
			
			# Arbitrary value for steering angle for left / right cam images
			correction = 0.3		

			for offset in (range(0, num_samples, batch_size)):
				processed_samples = processed_samples + batch_size
				batch_samples = samples[offset:offset+batch_size]
				images = []
				angles = []

				for batch_sample in batch_samples:

					name = data_path+'/IMG/'+batch_sample[0].split('/')[-1]
					center_image = cv2.imread(name)

					# As the Image frames are processed by drive.py is in RGB colorspace
					center_image = cv2.cvtColor(center_image,cv2.COLOR_BGR2RGB) 
					center_angle = float(batch_sample[3])
					images.append(center_image)
					angles.append(center_angle)

					# Augmentation of training images using flip
					if augment_images == True:
						center_image_flipped = np.fliplr(center_image)
						center_angle_flipped = -center_angle
						images.append(center_image_flipped)
						angles.append(center_angle_flipped)

					# Adding left and right camera images with correction
					if center_only==False:
						name = data_path+'/IMG/'+batch_sample[1].split('/')[-1]
						left_image = cv2.imread(name)
						left_image = cv2.cvtColor(left_image,cv2.COLOR_BGR2RGB) 
						left_angle = float(batch_sample[3]) + correction
						images.append(left_image)
						angles.append(left_angle)
						if augment_images == True:
							left_image_flipped = np.fliplr(left_image)
							left_angle_flipped = -left_angle
							images.append(left_image_flipped)
							angles.append(left_angle_flipped)

						name = data_path+'/IMG/'+batch_sample[2].split('/')[-1]
						right_image = cv2.imread(name)
						right_image = cv2.cvtColor(right_image,cv2.COLOR_BGR2RGB) 
						right_angle = float(batch_sample[3]) - correction
						images.append(right_image)
						angles.append(right_angle)
						if augment_images == True:
							right_image_flipped = np.fliplr(right_image)
							right_angle_flipped = -right_angle
							images.append(right_image_flipped)
							angles.append(right_angle_flipped)

				X_array = np.array(images)
				y_array = np.array(angles)
				yield shuffle(X_array, y_array)

			# To check whether all training samples are processed for a single epoch
			# processed_samples shouldn't exceed num_samples
			processed_samples = processed_samples-(processed_samples % num_samples)
			print("{} : {} samples of {} has been processed".format(generator_name, processed_samples,
																	num_samples))
			print("Augmentation : {}, Considered image from only center camera : {}".format(augment_images,
																						center_only))

	return generator_function

	

if __name__ == '__main__':
	
	# Set the path where csv and sample images are stored
	data_path = "../data/data/"
	
	# read the image paths and steering angles from csv file
	data_samples = fetch_data()

	# Split the samples into training and validation samples
	train_samples, validation_samples = train_test_split(data_samples, test_size=0.2)

	# Create generator functions that yields batchs for training and validation sets
	# with appropriate flags
	train_generator = generator(train_samples, batch_size=32, generator_name="Training set  ")
	validation_generator = generator(validation_samples, batch_size=32, generator_name="Validation set",
									center_only=True, augment_images=False)

	# Create a complied model 
	model = create_nvidia_model((160,320,3))

	# Define the batch size
	batch_size = 32

	# Execute training using fit_generator()
	history_object = model.fit_generator(train_generator(), steps_per_epoch = len(train_samples) / batch_size, 
										validation_data=validation_generator(),
										validation_steps=len(validation_samples) / batch_size, 
										epochs=10, verbose = 2)
									 
	# Save the model
	model.save("model.h5")
									 
