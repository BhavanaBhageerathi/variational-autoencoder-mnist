# Variational Autoencoder (VAE) using PyTorch

## Overview

This project implements a Variational Autoencoder (VAE) using PyTorch and trains it on the MNIST handwritten digit dataset.

A VAE is a generative deep learning model that learns compressed representations of input data in a latent space. It uses an encoder to map input images into a latent representation and a decoder to reconstruct images or generate new samples.

## Features

* Implemented an encoder and decoder using PyTorch.
* Learned latent representations using mean and log variance.
* Implemented the reparameterization trick for latent-space sampling.
* Used reconstruction loss and KL divergence for VAE loss calculation.
* Trained the model using the MNIST dataset.
* Generated new handwritten digit images by sampling random points from the latent space.
* Visualized generated images in a 4 × 4 grid.

## Technologies Used

* Python
* PyTorch
* Torchvision
* NumPy
* Matplotlib
* MNIST Dataset

## Model Architecture

The VAE consists of three main components:

1. **Encoder:** Compresses the input image into a hidden representation.
2. **Latent Space:** Learns the mean and log variance used to sample latent vectors.
3. **Decoder:** Reconstructs images from latent vectors.

### Model Configuration

* Input dimension: 784 (28 × 28 pixels)
* Hidden dimension: 400
* Latent dimension: 20
* Batch size: 64
* Optimizer: Adam
* Learning rate: 0.001
* Training epochs: 10

## Loss Function

The VAE uses two loss components:

* **Reconstruction Loss:** Measures how closely the reconstructed image matches the original input.
* **KL Divergence Loss:** Encourages the learned latent distribution to follow a standard normal distribution.

The total loss is the sum of these two components.

## Dataset

The project uses the MNIST handwritten digit dataset, containing grayscale images of handwritten digits from 0 to 9.

Each image has a dimension of 28 × 28 pixels.

The dataset is automatically downloaded using Torchvision when the program runs.

## Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/your-username/variational-autoencoder-mnist.git
```

### 2. Navigate to the project folder

```bash
cd variational-autoencoder-mnist
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

Windows:

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the project

```bash
python vae_mnist.py
```

The model trains on the MNIST dataset and displays 16 generated handwritten digit images in a 4 × 4 grid.

## Project Output

The trained VAE generates new handwritten digit images by sampling random vectors from the latent space and passing them through the decoder.

## Author

Bhavana Bhageerathi
