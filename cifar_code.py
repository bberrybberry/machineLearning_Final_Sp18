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
from keras.datasets import cifar10
from keras import applications
from keras import optimizers
from keras import initializers
from keras import utils
import pandas
import os
import time
import datetime
# dimensions of our images.
img_width, img_height = 32, 32

# model variables
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 1080*85
nb_validation_samples = 120*85
epochs = 50
batch_size = 256
nb_nodes = 4096
nb_nodes_last = 1000
nb_nodes_small_factor = 1 
data_augmentation = True

def buildVggA(num_classes):
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
	# TODO: For model A, I don't think that line above should be commented out, but I'll leave it as is for now
	
	#output softmax
	vggInspired.add(Dense(num_classes, activation='softmax'))
	
	vggInspired.summary()
	
	return vggInspired

def buildVggD(num_classes):
	#Resize arrays
	inputShape = (img_width, img_height, 3)
	
	# Create model based on VGG-D
	vggInspired = Sequential()
	
	#first conv layer
	vggInspired.add(Conv2D(64, kernel_size=3, strides=1, input_shape=inputShape, padding='same', activation='relu'))
	vggInspired.add(Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu'))
	vggInspired.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
	
	#second conv layer
	vggInspired.add(Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu'))
	vggInspired.add(Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu'))
	vggInspired.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
	
	#third conv layer
	vggInspired.add(Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu'))
	vggInspired.add(Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu'))
	vggInspired.add(Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu'))
	vggInspired.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
	
	#fourth conv layer
	vggInspired.add(Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'))
	vggInspired.add(Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'))
	vggInspired.add(Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'))
	vggInspired.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
	
	#fifth conv layer
	vggInspired.add(Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'))
	vggInspired.add(Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'))
	vggInspired.add(Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'))
	vggInspired.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
	
	#flatten
	vggInspired.add(Flatten())
	
	randnorm = initializers.RandomNormal(mean=0.0, stddev=0.1)

	#fc layers
	vggInspired.add(Dense(nb_nodes // nb_nodes_small_factor, kernel_initializer=randnorm))
	vggInspired.add(Dropout(.5))
	vggInspired.add(Dense(nb_nodes // nb_nodes_small_factor, kernel_initializer=randnorm))
	vggInspired.add(Dropout(.5))
	vggInspired.add(Dense(nb_nodes_last // nb_nodes_small_factor))
	
	#output softmax
	vggInspired.add(Dense(nb_classes, activation='softmax'))
	
	vggInspired.summary()
	
	return vggInspired
	
def trainSimpleVgg():
	# Load data
	num_classes = 10
	(x_train, y_train), (x_test, y_test) = cifar10.load_data()
	y_train = utils.to_categorical(y_train, num_classes)
	y_test = utils.to_categorical(y_test, num_classes)

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255

	
	model = buildVggA(num_classes)

	# Set optmizer and compile model
	learning_rate = .001
	sgd = optimizers.SGD(lr=learning_rate, momentum=0.9) #if we're still doing bad with lr ~ .000001, then give up. Persist until then
	adam = optimizers.Adam(lr=learning_rate)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])
	
	# Prepare model model saving directory.
	save_dir = os.path.join(os.getcwd(), 'saved_models')
	model_name = 'vgg16_cifar.{epoch:03d}.h5'
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)
	filepath = os.path.join(save_dir, model_name)

	# Save train models and fit generator
	checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=1, save_best_only=True)
	callbacks = [checkpoint]	
	h = 0
	if not data_augmentation:
		h = model.fit(
			x_train,
			y_train,
			batch_size=batch_size,
			epochs=epochs,
			validation_data=(x_test, y_test),
			callbacks=callbacks,
			verbose=1)
	else:
		datagen = ImageDataGenerator(
			featurewise_center=True,
			featurewise_std_normalization=True,
			rotation_range=20,
			width_shift_range=0.2,
			height_shift_range=0.2,
			shear_range=0.2,
			zoom_range=0.2,
			horizontal_flip=True)

		datagen.fit(x_train)

		h = model.fit_generator(
			datagen.flow(x_train, y_train, batch_size=batch_size),
			epochs=epochs,
			validation_data=(x_test, y_test),
			workers=4)


	return h

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S')
print(st)

# Run
history = trainSimpleVgg()

# Save
pandas.DataFrame(history.history).to_csv("history" + st + ".csv")
