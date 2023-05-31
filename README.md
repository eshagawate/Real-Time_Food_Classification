# Real-Time_Food_Classification

## Introduction:

InceptionV3 is a popular deep learning model that has shown excellent performance in image classification tasks. InceptionV3 is a deep convolutional neural network (CNN) architecture developed by Google. It is an extension of the earlier Inception models and has been widely used for various image classification tasks. The architecture is characterized by its unique "Inception module" design, which allows the network to capture information at different scales and resolutions efficiently.

## The key features of the InceptionV3 architecture include:

1. Multiple parallel convolutional pathways: The Inception module consists of convolutional layers with different filter sizes (1x1, 3x3, and 5x5). These parallel pathways are designed to capture features at different spatial scales and provide a richer representation of the input.

2. Dimensionality reduction: InceptionV3 incorporates 1x1 convolutions as bottleneck layers to reduce the dimensionality of the input. This helps reduce computational complexity while preserving important features.

3. Auxiliary classifiers: InceptionV3 includes auxiliary classifiers at intermediate layers of the network, which are used during training to provide additional supervision. These auxiliary classifiers help combat the vanishing gradient problem and improve the flow of gradients during backpropagation.

4. Global average pooling: Instead of using fully connected layers at the end, InceptionV3 employs global average pooling, which reduces the spatial dimensions of the feature maps to a single vector. This approach reduces overfitting and makes the model more robust to variations in input size.

## Benefits of using InceptionV3 for food classification:

InceptionV3 offers several benefits when applied to food classification tasks:

High accuracy: InceptionV3 has demonstrated impressive performance on large-scale image classification benchmarks like ImageNet. It has been trained on a diverse range of objects, including food items, enabling it to learn discriminative features for accurate food classification.

Efficient feature extraction: The InceptionV3 architecture's multi-scale feature extraction capabilities make it effective at capturing fine-grained details and global context in food images. This is particularly valuable for distinguishing visually similar food categories.

Transfer learning capability: InceptionV3 is typically pre-trained on large-scale datasets like ImageNet, which exposes it to a wide variety of images. This pre-training allows the model to learn general visual representations that can be fine-tuned for specific tasks like food classification with relatively small amounts of labeled data.

Understanding the concept of transfer learning:
Transfer learning is a machine learning technique where knowledge gained from training one model on a source task is leveraged to improve the performance of a related target task. In the context of deep learning, it involves using a pre-trained model as a starting point for a new task instead of training from scratch.

By utilizing transfer learning with InceptionV3, we can benefit from the features learned on large-scale datasets such as ImageNet. The early layers of InceptionV3 learn low-level features like edges and textures, while the deeper layers capture high-level concepts and semantic information. These features generalize well to various visual recognition tasks, including food classification.

The code contains following sections:

## Dataset Preparation:

Gathering a food dataset for training and evaluation
Annotating and labeling the images with appropriate food categories
Splitting the dataset into training and testing sets

![image](https://github.com/eshagawate/Real-Time_Food_Classification/assets/115074194/c2a5e180-fdec-4dc2-b4ce-0defbce54bba)

## Fine-tuning InceptionV3:

Loading the pre-trained InceptionV3 model with ImageNet weights
Modifying the last few layers for food classification
Freezing and unfreezing specific layers for fine-tuning
Choosing an appropriate loss function and optimizer

## Data Augmentation:

Augmenting the training dataset to increase its size and diversity
Techniques such as random rotations, flips, and zooming
Improving the model's generalization capabilities

## Training the Model:

Preparing the input data for feeding into the model
Training the fine-tuned InceptionV3 model on the food dataset
Monitoring the training process and evaluating the model's performance

## Real-Time Food Classification:

Loading the trained InceptionV3 model
Preprocessing the input image for classification
Utilizing the forward pass of the model for prediction
Extracting the predicted class and its probability

![Screenshot 2023-05-30 161306](https://github.com/eshagawate/Real-Time_Food_Classification/assets/115074194/5428623e-4ab2-4fb1-88ba-fd5854c965ac)

![Screenshot 2023-06-01 004540](https://github.com/eshagawate/Real-Time_Food_Classification/assets/115074194/d0456f0a-7cdf-4ae1-9afe-95d5017e501c)

## Generating bill for the users mentioning the calorie count

![image](https://github.com/eshagawate/Real-Time_Food_Classification/assets/115074194/78f3c524-da40-4780-9f15-fa83c7216a16)


