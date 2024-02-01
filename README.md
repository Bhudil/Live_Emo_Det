# Real-Time Emotion Detection

## Overview

The provided code implements an Emotion Detection system using deep learning techniques for image analysis. It utilizes a Convolutional Neural Network (CNN) architecture built with the Keras framework and TensorFlow backend to classify facial expressions into seven emotional categories: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral. The model is trained on a dataset of grayscale facial images, and its training progress is monitored using callbacks for early stopping and model checkpointing. The code also includes a real-time emotion detection application using OpenCV, where the trained model is applied to classify emotions in live video feed from a webcam, with faces detected using Haar cascades. Also provies output of the number of faces detected in the live stream. The project demonstrates the integration of machine learning, computer vision, and deep learning to create an interactive and responsive emotion recognition system.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Dataset](#dataset)
- [Installation](#installation)
- [Model Architecture](#model-architecture)
- [Results](#results)

## Prerequisites

Before you begin, ensure you have the following dependencies installed:

- Python 3.x
- TensorFlow
- Keras
- OpenCV
- Matplotlib
- NumPy
- Pandas

You can install the required packages using the following command:

```bash
pip install -r requirements.txt
```

## Dataset 

The used dataset is taken from kaggle at: [DATASET](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset)

## Installation

```bash
git clone https://github.com/Bhudil/Emotion_Detection.git
cd Emotion_Detection
```

## Model Architecture

The emotion recognition model uses a deep CNN architecture with the following layers:

Convolutional layers with batch normalization and ReLU activation
Max pooling layers
Dropout layers for regularization
Dense layers with batch normalization and ReLU activation
Output layer with softmax activation
The model is compiled using the Adam optimizer and categorical crossentropy loss.

## Results

**Single Test (#1) :**


**Single Test (#2) :**


**Multi Test (#1) :**


**Multi Test (#1) :**
