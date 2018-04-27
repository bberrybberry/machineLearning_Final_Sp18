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

# dimensions of our images.
img_width, img_height = 100, 100

# model variables
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 200
nb_validation_samples = 80
epochs = 10
batch_size = 10
nb_classes = 2


def trainSimpleVgg():
	# load data
	    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=.2,
            zoom_range=.2,
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
	nb_classes = train_gen.classes

    #Resize arrays
    inputShape = (img_width, img_height, 3)
    
    # Create model based on VGG-A
    vggInspired = Sequential()
    
    #first conv layer
    vggInspired.add(Conv2D(64, kernel_size=3, strides=1, input_shape=inputShape, padding='same', activation='relu'))
    vggInspired.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
    
    #second conv layer
    vggInspired.add(Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu'))
    vggInspired.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
    
    #third conv layer
    vggInspired.add(Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu'))
    vggInspired.add(Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu'))
    vggInspired.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
    
    #fourth conv layer
    vggInspired.add(Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'))
    vggInspired.add(Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'))
    vggInspired.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
    
    #fifth conv layer
    vggInspired.add(Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'))
    vggInspired.add(Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'))
    vggInspired.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
    
    #flatten
    vggInspired.add(Flatten())
    
    #fc layers
    vggInspired.add(Dense(4096))
    vggInspired.add(Dense(4096))
    vggInspired.add(Dense(4096))
    
    #output softmax
    vggInspired.add(Dense(nb_classes, activation='softmax'))
    
    vggInspired.summary()
    vggInspired.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=["accuracy"])
    

    
    h = vggInspired.fit_generator(
            train_gen,
            steps_per_epoch=nb_train_samples/batch_size,
            epochs=epochs,
            validation_data=test_gen,
            validation_steps=nb_validation_samples,
            verbose=1)
    return h


history = trainSimpleVgg()