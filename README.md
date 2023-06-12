# Variational Autoencoders

![Variational Autoencoders](vae.png)

This repository contains an implementation of Variational Autoencoders (VAEs), a popular type of generative deep learning model. Variational Autoencoders are a powerful tool for unsupervised learning, dimensionality reduction, and generating new data samples.

## Table of Contents

- [Introduction to Variational Autoencoders](#introduction-to-variational-autoencoders)
- [How VAEs Work](#how-vaes-work)
- [Repository Structure](#repository-structure)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)


## Introduction to Variational Autoencoders

Variational Autoencoders are a type of generative model that combines ideas from traditional autoencoders and Bayesian inference. They are capable of learning a compressed representation of the input data and generating new data samples similar to the training data.

The key idea behind VAEs is to learn a latent space representation of the input data that follows a specific probability distribution, usually a multivariate Gaussian distribution. By imposing this distribution on the latent space, VAEs enable the generation of new data samples by sampling from the latent space and decoding the samples back into the original data space.

## How VAEs Work

1. **Encoder**: The encoder network takes the input data and maps it to the latent space. It consists of several layers that progressively reduce the dimensionality of the input and output the mean and variance parameters of the latent space distribution.

2. **Latent Space Sampling**: From the mean and variance parameters output by the encoder, we sample a point in the latent space using the reparameterization trick. This sampling process introduces stochasticity into the model, allowing for the generation of diverse outputs.

3. **Decoder**: The decoder network takes the sampled point from the latent space and reconstructs the input data. It consists of several layers that progressively upsample and refine the data until the final output is generated.

4. **Loss Function**: The VAE is trained by minimizing a loss function that consists of two components: reconstruction loss and regularization loss. The reconstruction loss measures the similarity between the input and the reconstructed output, while the regularization loss encourages the latent space distribution to follow the desired probability distribution.

5. **Training**: During training, the VAE is optimized using gradient descent to minimize the loss function. The training process updates the parameters of both the encoder and decoder networks, enabling them to learn a meaningful latent space representation and generate new data samples.

## Repository Structure

This repository has the following structure:

```
|- variational auto encoder.ipynb              # Jupyter Notebook with VAE implementation and examples
|- Updated vae.ipynb      # High Level and more generalized implememtation with example
|- requirements.txt       # List of dependencies
|- README.md              # Project Readme file
```

## Dependencies

The implementation requires the following dependencies:

- Python (3.6+)
- TensorFlow (2.0+)
- NumPy
- Matplotlib
- Jupyter notebook

You can install the required packages by running the following command:

```
pip install -r requirements.txt
```

## Usage

To use the VAE implementation in this repository, follow these steps:

1. Download your dataset: Download your dataset using tensorflow inbuilt methods.

2. Configure the VAE settings: Modify the hyperparameters and network architecture in the `variational auto encoder.ipynb` file to suit your needs.

 You can adjust the model architecture, learning rate, batch size, latent space dimension, etc.

3. Train the VAE: Run the `variational auto encoder.ipynb` script to train the VAE on your dataset.

4. Generate new samples: Use the trained VAE to generate new data samples by changing the values of scale parameter 

## Examples

The `variational auto encoder.ipynb` notebook in this repository provides examples of using the VAE implementation. It demonstrates how to train a VAE on a specific dataset and generate new samples using the trained model.

## Contributing

Contributions to this repository are always welcome. If you find any issues or have suggestions for improvements, please create an issue or submit a pull request.

# Cheers

P.S. - If you face any error regarding the output try to run model again on google colab
