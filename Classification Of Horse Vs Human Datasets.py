import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# specify the paths to your training and validation directories
train_dir = 'C:/Users/sunil/Documents/GitHub/Deep-Feedforward-neural-network-in-Keras/Training'

# Training/
# │
# ├── horses/
# │   ├── horse01.jpg
# │   ├── horse02.jpg
# │   └── ...
# │
# └── humans/
#     ├── human01.jpg
#     ├── human02.jpg
#     └── ...

validation_dir = 'C:/Users/sunil/Documents/GitHub/Deep-Feedforward-neural-network-in-Keras/Validation'
# validation_dir/
# │
# ├── horses/
# │   ├── horse01.jpg
# │   ├── horse02.jpg
# │   └── ...
# │
# └── humans/
#     ├── human01.jpg
#     ├── human02.jpg
#     └── ...

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(150, 150, 3)),  # first, flatten the images
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # binary output layer
])

model.compile(optimizer=RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 128
train_generator = train_datagen.flow_from_directory(
        train_dir,  
        target_size=(150, 150), 
        batch_size=128,
        class_mode='binary')

# Flow validation images in batches of 128
validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=128,
        class_mode='binary')

history = model.fit(
      train_generator,
      steps_per_epoch=8,  
      epochs=15,
      verbose=1,
      validation_data=validation_generator,
      validation_steps=8)
