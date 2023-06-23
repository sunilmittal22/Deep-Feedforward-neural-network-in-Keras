# The Fashion-MNIST dataset is a dataset of Zalando's article images, intended to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms. While the original MNIST dataset contains images of handwritten digits, the Fashion-MNIST dataset contains images of clothing items.

# The dataset contains 70,000 images (60,000 training examples and 10,000 test examples). Each example is a 28x28 grayscale image, associated with a label from 10 classes. The class labels are as follows:

# T-shirt/top
# Trouser
# Pullover
# Dress
# Coat
# Sandal
# Shirt
# Sneaker
# Bag
# Ankle boot
# Each pixel-value is an integer between 0 and 255, indicating grayscale level. It is good practice to normalize these values to a range of [0, 1] before feeding into a neural network model, as this can help with the training process.

# The Fashion-MNIST dataset is often used as a beginner's dataset for image classification problems, as it's more complex than the MNIST digit dataset, but not as large or complex as some other image datasets like CIFAR-10 or ImageNet.

# Import the necessary libraries
from tensorflow.keras.datasets import fashion_mnist  # The dataset we'll be using
from tensorflow.keras.models import Sequential  # The type of model we'll be using
from tensorflow.keras.layers import Dense, Flatten  # The layers we'll be using in our model
from tensorflow.keras.utils import to_categorical  # A utility function to convert labels into one-hot format

# Load the Fashion MNIST dataset from keras
# The line of code you've provided is used to load the Fashion-MNIST dataset from the fashion_mnist module in Keras.

# Here's a breakdown:

# fashion_mnist.load_data(): This function call loads the Fashion-MNIST dataset. This dataset is included with Keras and can be accessed using this function. The dataset comes pre-split into a training set and a test set.

# (train_images, train_labels), (test_images, test_labels) = : This is Python's multiple assignment feature, and it's being used here to assign the data returned by fashion_mnist.load_data() to variables. The load_data() function returns two tuples:

# The first tuple contains the training images and their corresponding labels (train_images and train_labels).
# The second tuple contains the testing or validation images and their corresponding labels (test_images and test_labels).
# After this line of code is run, you have four variables:

# train_images is a numpy array of shape (60000, 28, 28) containing the training images.
# train_labels is a numpy array of shape (60000,) containing labels for the training images.
# test_images is a numpy array of shape (10000, 28, 28) containing the testing images.
# test_labels is a numpy array of shape (10000,) containing labels for the testing images.
# Each image is represented as a 28x28 pixel grayscale image, and the labels are integer labels representing the class of clothing the image represents.

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize the pixel values of the images to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Convert the labels (integers) into one-hot vectors
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# ReLU stands for Rectified Linear Unit, and it is a type of activation function commonly used in neural networks and deep learning models. The function returns 0 if the input is less than 0, and it returns the input itself if the input is equal to or greater than 0.

# Mathematically, it can be represented as:

# scss
# Copy code
# f(x) = max(0, x)
# Where x is the input to a neuron.

# This means that the ReLU function is linear for all positive values, and is zero for all negative values. It is a non-linear function overall (since it isn't linear for negative values), which is important because this non-linearity helps neural networks learn from complex data.

# The ReLU activation function has become very popular for several reasons:

# It helps mitigate the vanishing gradient problem, which is a challenge when training deep neural networks.
# It introduces non-linearity into the network without requiring computationally expensive operations like exponentials or cosines.
# Its simplicity allows the network to learn much faster.
# However, ReLU units can be fragile during training and can "die". For example, a large gradient flowing through a ReLU neuron can cause the weights to update in such a way that the neuron will never activate on any datapoint again. If this happens, then the gradient flowing through the unit will forever be zero from that point on. This is often referred to as the "dying ReLU" problem. Some variants like the Leaky ReLU or Parametric ReLU have been proposed to mitigate this problem.

# The softmax function, also known as the normalized exponential function, is a function that takes as input a vector of K real numbers, and normalizes it into a probability distribution consisting of K probabilities proportional to the exponentials of the input numbers.

# The output of the softmax function is a vector that represents the probability distributions of a list of potential outcomes. It's most commonly used in applied machine learning, where it serves as an activation function in a neural network model, often being used in the output layer of a classifier model.

# Mathematically, it is defined as follows:

# scss
# Copy code
# softmax(x_i) = exp(x_i) / Σ(exp(x_j)) for j from 1 to K
# Where:

# x is the input vector to the softmax function (x_1, x_2, ..., x_K), and
# exp(x) is the exponential function.
# Key properties of the softmax function:

# Range: The output of softmax is a vector that represents the probability distributions of a list of potential outcomes. Each number in this vector falls in the range (0, 1), and the total sum of the probabilities is equal to 1.

# Monotonicity: The softmax function is monotonic, meaning that if one component of the input vector is larger than another, that relationship is preserved in the output probability distribution.

# High sensitivity: Softmax amplifies the impact of the maximum input value and suppresses all values which are significantly lower than the maximum one. This makes the model output quite sensitive to the maximum input value and less sensitive to others.

# These properties make it a good choice for multi-class classification problems, as it allows the model to output a probability distribution over classes.

# Define our model architecture
# Sequential is a class in Keras that allows us to construct neural networks in a very intuitive and easy way. It lets you easily create models layer-by-layer.

# When creating a neural network in Keras, you generally have two options:

# Sequential model: This is the simplest kind of neural network model in Keras. It's called 'sequential' because it allows you to sequentially stack layers in a neural network, where each layer has exactly one input tensor and one output tensor. In other words, you are creating a linear stack of layers where each layer only feeds into the next.

# Functional API: For more complex models, like multi-output models, directed acyclic graphs, or models with shared layers, Keras offers the Functional API.

# Here's an example of how you might use Sequential:

# python
# Copy code
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

# # Instantiate a Sequential model
# model = Sequential()

# # Add a Dense layer with 10 neurons and 'relu' activation function
# model.add(Dense(10, activation='relu'))

# # Add another Dense layer with 5 neurons and 'relu' activation function
# model.add(Dense(5, activation='relu'))

# # Add a final Dense layer with 1 neuron and 'sigmoid' activation function
# model.add(Dense(1, activation='sigmoid'))
# In the example above, an instance of the Sequential model is created. Then three layers are added using the add() function. The model has a simple linear architecture: the input flows into the first layer, then into the second, and so forth until it reaches the output layer.

# Remember, the Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor, but it doesn't handle models with multiple inputs or outputs, which are better handled with the Keras functional API.
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Input layer: flattens the 28x28 pixel image into a 784 element array
    Dense(512, activation='relu'),  # Hidden layer: 512 neurons, ReLU activation
    Dense(256, activation='relu'),  # Hidden layer: 256 neurons, ReLU activation
    Dense(128, activation='relu'),  # Hidden layer: 128 neurons, ReLU activation
    Dense(10, activation='softmax')  # Output layer: 10 neurons (for the 10 classes), Softmax activation
])
# The softmax activation function is typically used in the output layer of a neural network model for multi-class classification problems.

# There are a few key reasons why softmax is used in this context:

# Probabilistic Outputs: Softmax converts raw output scores from the previous layer into probabilities for each class. This is particularly useful in classification problems where we're interested in knowing the probabilities of each class.

# Multi-class Classification: Unlike functions like the sigmoid function, which are used for binary classification, softmax is useful for multi-class classification. This is because it generates a probability distribution over K different possible outcomes, making it perfect for representing categorical outcomes.

# Differentiability: The softmax function is differentiable, which means we can calculate derivatives. This is a critical property that lets us use backpropagation to train the neural network.

# Highlighting the Maximum Input: The softmax function tends to highlight the largest values and suppress values which are significantly below the maximum one. This characteristic can be very useful when we want to identify a clear choice out of a set of options.

# For example, let's say you have a neural network model for image classification with 10 output nodes, one for each digit from 0 to 9. In the output layer, you would use the softmax activation function, so that the output of each node is a probability that the image belongs to that particular class, and the sum of the outputs is 1. This makes the outputs of the model easy to interpret as confidence scores for each class.
# Compile the model. We use Adam as our optimizer, categorical cross entropy as our loss function, 
# and accuracy as the metric to track during training.
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model for 10 epochs using a batch size of 32. The model learns the optimal weights and biases during this process.
model.fit(train_images, train_labels, epochs=10, batch_size=32)

# Evaluate the model on the test data. This gives us a measure of how well the model performs on unseen data.
test_loss, test_acc = model.evaluate(test_images, test_labels)

# Print out the accuracy of the model on the test data
print('Test accuracy:', test_acc)
