Pix2PixHD model
=============
Solar farside magnetograms are generated by the Pix2PixHD model using pairs of SDO/AIA EUV passband images and SDO/HMI LOS magnetograms.   


Network architectures
-------------
The model consists of two major networks: one is a generative network (generator) and the other is a discriminative network (discriminator).
The generator tries to generate realistic output from input, and the discriminator tries to distinguish which one is a more real-like pair between a real pair and a fake pair.  

__Generator architectures__

In our model, we use a global generator.
The generator is consist of the Encoder - Residual Blocks - Decoder.

* Encoder


* Residual Blocks


* Decoder



__Discriminator architectures__

In our model, we use two 70*70 patch discriminator.
One discriminator gets input pairs of the original pixel size, and the other gets input pairs which are downsampled by half.


Hyperparameters
-------------

__The Loss configuration of the Objective functions__
* Total loss = ( cGAN loss ) + 10 * ( Feature Matching loss )   

__Optimizer__
* Optimizer : Adam solver
* Learning rate : 0.0002
* momentum beta 1 parameter : 0.5
* momentum beta 2 parameter : 0.999   

__Initializer__
* Initialize Weights in Convolutional Layers : normal distribution, mean : 0.0, standard deviation : 0.02   
