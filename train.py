# import the needed libraries
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Activation
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

# config
img_width, img_height = 28,28 #width & height of input image
input_depth = 1 #1: gray image
epochs = 50 #number of training epoch
batch_size = 15 #training batch size

base_dir = 'data'

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    zoom_range=0.1,
    validation_split=0.2 # 20% de los datos se usan para validacion
    )

train_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(28, 28),
    color_mode='grayscale',
    shuffle=True,
    batch_size=batch_size,
    class_mode= 'categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(28, 28),
    color_mode='grayscale',
    shuffle=True,
    batch_size=batch_size,
    class_mode= 'categorical',
    subset='validation'
)



#define input image order shape
if K.image_data_format() == 'channels_first':
    input_shape_val = (input_depth, img_width, img_height)
else:
    input_shape_val = (img_width, img_height, input_depth)

#define the network
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'valid'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Flatten())

model.add(Dense(64,activation ="relu"))
model.add(Dense(128,activation ="relu"))

model.add(Dense(train_generator.num_classes,activation ="softmax"))



model.compile(optimizer = Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])


# Show the model summary
model.summary()    




history = model.fit(
    train_generator,#our training generator
    #number of iteration per epoch = number of data / batch size
    steps_per_epoch=np.floor(train_generator.n/batch_size),
    epochs=epochs,#number of epoch
    #callbacks=[reduce_lr, early_stop],
    validation_data=validation_generator,#our validation generator
    #number of iteration per epoch = number of data / batch size
    validation_steps=np.floor(validation_generator.n / batch_size))

print("Entrenamiento completado")
model.save('./model2/model15.h5')
print("Archivo guardado")    