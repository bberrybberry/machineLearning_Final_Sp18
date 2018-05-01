'''
	Dog breed classifer based on VVG16-A architecture.
	
	Assume the following directory structure:
		data/
			train/
				breed1/
					i1.jpg
					i2.jpg
					...
				breed2/
					i1.jpg
					i2.jpg
					...
				...
			validation/
				breed1/
					i1.jpg
					i2.jpg
					...
				breed2/
					i1.jpg
					i2.jpg
					...
				...
'''
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras import applications
from keras import optimizers
from keras import initializers
import pandas
import os

# dimensions of our images.
img_width, img_height = 100, 100

# model variables
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 1080*85
nb_validation_samples = 120*85
epochs = 100
batch_size = 120
nb_nodes = 4096
nb_nodes_last = 1000
nb_nodes_small_factor = 2

def trainSimpleVgg():
	# load data
	train_datagen = ImageDataGenerator(
			featurewise_center=True,
			featurewise_std_normalization=True,
			rescale=1./255,
			rotation_range=20,
			width_shift_range=0.2,
			height_shift_range=0.2,
			shear_range=0.2,
			zoom_range=0.2,
			horizontal_flip=True)
	test_datagen = ImageDataGenerator(rescale=1./255)
	
	train_gen = train_datagen.flow_from_directory(
			train_data_dir,
			target_size=(img_width, img_height),
			batch_size=batch_size,
			shuffle=False)
	test_gen = test_datagen.flow_from_directory(
			validation_data_dir,
			target_size=(img_width, img_height),
			batch_size=batch_size,
			shuffle=False)
	nb_classes = train_gen.num_classes

	#Resize arrays
	inputShape = (img_width, img_height, 3)
	
	# Create model based on VGG-D
	vggInspired = Sequential()
	
	#first conv layer
	vggInspired.add(Conv2D(64, kernel_size=3, strides=1, input_shape=inputShape, padding='same', activation='relu'))
	#vggInspired.add(Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu'))
	vggInspired.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
	
	#second conv layer
	vggInspired.add(Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu'))
	#vggInspired.add(Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu'))
	vggInspired.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
	
	#third conv layer
	vggInspired.add(Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu'))
	vggInspired.add(Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu'))
	#vggInspired.add(Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu'))
	vggInspired.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
	
	#fourth conv layer
	vggInspired.add(Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'))
	vggInspired.add(Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'))
	#vggInspired.add(Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'))
	vggInspired.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
	
	#fifth conv layer
	vggInspired.add(Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'))
	vggInspired.add(Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'))
	#vggInspired.add(Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'))
	vggInspired.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
	
	#flatten
	vggInspired.add(Flatten())
	
	randnorm = initializers.RandomNormal(mean=0.0, stddev=0.1)

	#fc layers
	vggInspired.add(Dense(nb_nodes // nb_nodes_small_factor, kernel_initializer=randnorm))
	vggInspired.add(Dropout(.5))
	vggInspired.add(Dense(nb_nodes // nb_nodes_small_factor, kernel_initializer=randnorm))
	vggInspired.add(Dropout(.5))
	#vggInspired.add(Dense(nb_nodes_last // nb_nodes_small_factor))
	
	#output softmax
	vggInspired.add(Dense(nb_classes, activation='softmax'))
	
	vggInspired.summary()

	sgd = optimizers.SGD(lr=0.01, momentum=0.9)
	# set optmizer and compile model
	vggInspired.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])
	
	# Prepare model model saving directory.
	save_dir = os.path.join(os.getcwd(), 'saved_models')
	model_name = 'vgg16_imagenet.{epoch:03d}.h5'
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)
	filepath = os.path.join(save_dir, model_name)

	checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=1, save_best_only=True)
	callbacks = [checkpoint]	
	h = vggInspired.fit_generator(
			train_gen,
			steps_per_epoch=nb_train_samples/batch_size,
			epochs=epochs,
			validation_data=test_gen,
			validation_steps=nb_validation_samples / batch_size,
			callbacks=callbacks,
			verbose=1)
	return h


history = trainSimpleVgg()

pandas.DataFrame(history.history).to_csv("history.csv")
