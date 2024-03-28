# Colorize-grayscale-images

This project implements a convolutional autoencoder (CAE) model to colorize black and white images using TensorFlow and Keras. The code utilizes the CIFAR-10 image dataset and leverages the CIELAB color space for color representation.

CIELAB Color Space: A perceptually uniform color space where L represents lightness, and a* and b* represent color dimensions. This color space is well-suited for colorization tasks.

# Dependencies

Python 3.x
NumPy
TensorFlow (2.x recommended)
Keras
matplotlib
scikit-image (skimage)

# Dataset

CIFAR-10 (https://www.cs.toronto.edu/~kriz/cifar.html)

# Installation and Execution

Install dependencies:

Bash
pip install numpy tensorflow keras matplotlib scikit-image
Use code with caution.

Download the CIFAR-10 Dataset (if not already present):
You can often download this dataset automatically within the code using keras.datasets.cifar10.load_data().

# Run the code:

Bash
python colorization_code.py 


# Code Structure

Data Loading and Preprocessing: Load the CIFAR-10 dataset, display sample images, and convert images into the CIELAB color space. Images are split into training and testing sets.
Model Definition: Define the CAE architecture with convolutional and upsampling layers.
Compilation: Choose an optimizer ('adam') and loss function ('mse') appropriate for this regression task.
Training: Train the CAE model on the grayscale images (L channel) to predict the color channels (a* and b*).
Colorization: Use the trained model to colorize grayscale images from the test set.
Visualization: Visualize the original color image, the grayscale image, and the colorized image.

# Notes

Experiment with different CAE architectures (e.g., deeper layers, more filters).
Explore other color spaces or colorization techniques.
Evaluate the colorized images with appropriate metrics
