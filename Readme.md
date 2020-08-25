Pix2PixHD model for the generation of the solar farside magnetograms
=============

Network architectures
-------------
The model consists of two major networks: one is a generative network (generator, G) and the other is a discriminative network (discriminator, D).
The generator tries to generate realistic output from input, and the discriminator tries to distinguish which one is a more real-like pair between a real pair and a fake pair.


Hyperparameters
-------------

The loss configuration of the objective functions of the Generator
* cGAN loss and feature matching loss

Optimizer
* Optimizer : Adam solver
* Learning rate : 0.0002
* momentum beta 1 parameter : 0.5
* momentum beta 2 parameter : 0.999

Initializer
* Initialize Weights in Convolutional and Transposed Convolutional Layers : normal distribution, mean : 0.0, standard deviation : 0.02
