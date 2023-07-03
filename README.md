# Image-processing-

In the scope of this project, we investigated the performance of 4 models, namely Multi-layer Perceptron
(MLP), Convolution Neural Networks (CNN), and two pre-trained models, Resnet50 and Densenet201,
on the CIFAR-10 dataset. MLP with two hidden layers ( RelU activation function each), with a learning
rate of 0.01, a batch size of 64, as well as L2 regularization, achieves a modest performance of around
51.36%. Moreover, the CNN model with a learning rate of 0.001 converged to its optimal point after 15
epochs reaching a testing accuracy of 68.95 %. Data augmentation techniques and Dropout method were
performed for the pre-trained models as well. Resnet 50 revealed a good testing performance reaching
around 84%. Finally, the DensNet201 reached the highest accuracy 88% with a learning rate of 0.0001
using Adam optimizer.
1 Introduction
The goal of this project is to investigate the performance of the Multi-layer perceptron model while varying
its various hyperparameters on the CIFAR-10 dataset. We also want to highlight the performance of the
CNN and pre-trained models ResNet and DenseNet on the same dataset. We found that the pre-trained
model DenseNet has significantly outperformed the other models reaching 88% of evaluation accuracy. The
CIFAR-10 dataset is commonly used as a benchmark for machine learning algorithms and deep learning
models in computer vision research. CIFAR-10 was used in [1] to demonstrate the effectiveness of ResNet
in handling the problem of vanishing gradients in very deep neural networks. It was also used in [2] to
evaluate the performance of four Convolutional Neural Network (CNN) models for image recognition and
classification. In [3], the CIFAR-10 dataset was used to demonstrate that some common regularization
techniques (such as weight decay and dropout) may not always improve generalization performance.
2 Dataset
The CIFAR-10 dataset is a well-known computer vision dataset consisting of 60,000 32x32 color images
in 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck), with 6,000 images
per class. The dataset is divided into 50,000 training images and 10,000 test images.
The images are relatively low resolution compared to some other datasets, but the variety of objects and
backgrounds, as well as a large number of images, make it a popular choice for image classification tasks.
We used a bar chart to visualize the distribution of the 10 classes, which shows the same number of images
(6,000) for each class, indicating a balanced dataset. We also visualized different images for each class in
order to get a sense of the different objects of the dataset. The image data must be processed to remove
unwanted distortions or enhance certain image features that are important for our model to perform better.
1
