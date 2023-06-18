import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.config.run_functions_eagerly(True)
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

train_dataset = tf.data.Dataset.from_generator(
    lambda: train_generator,
    output_signature=(
        tf.TensorSpec(shape=(32, 150, 150, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(32,), dtype=tf.int32))
).repeat()

validation_dataset = tf.data.Dataset.from_generator(
    lambda: validation_generator,
    output_signature=(
        tf.TensorSpec(shape=(32, 150, 150, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(32,), dtype=tf.int32))
).repeat()

# determine the number of steps per epoch
steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps = validation_generator.samples // validation_generator.batch_size

# history = model.fit(
#       train_generator,
#       steps_per_epoch=8,  
#       epochs=15,
#       verbose=1,
#       validation_data=validation_generator,
#       validation_steps=8)
# train the model

history = model.fit(train_dataset,
                    validation_data=validation_dataset,
                    steps_per_epoch=steps_per_epoch,
                    validation_steps=validation_steps,
                    epochs=10,
                    verbose=1
                    )