Pix2PixHD model
=============
Solar farside magnetograms are generated by the Pix2PixHD model using pairs of SDO/AIA EUV passband images and SDO/HMI LOS magnetograms.   


Network architectures
-------------
The model consists of two major networks: one is a generative network (generator, G) and the other is a discriminative network (discriminator, D).
The generator tries to generate realistic output from input, and the discriminator tries to distinguish which one is a more real-like pair between a real pair and a fake pair.   



Hyperparameters
-------------

__The Loss configuration of the Objective functions of the Generator__
* cGAN loss and feature matching loss   

__Optimizer__
* Optimizer : Adam solver
* Learning rate : 0.0002
* momentum beta 1 parameter : 0.5
* momentum beta 2 parameter : 0.999   

__Initializer__
* Initialize Weights in Convolutional and Transposed Convolutional Layers : normal distribution, mean : 0.0, standard deviation : 0.02   
