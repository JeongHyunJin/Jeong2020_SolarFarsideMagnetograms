Generation of solar farside magnetograms
=============
Solar farside magnetograms are generated by the Pix2PixHD model using pairs of SDO/AIA EUV passband images and SDO/HMI LOS magnetograms.   
<p align="center">
<img src="https://user-images.githubusercontent.com/68056295/91625647-f2e28180-e9e3-11ea-8469-f9d4c932fe35.png" width="70%" height="70%"></center>
</p>

Pix2PixHD model
===============
_____________  
   Network architectures
-------------
      The model consists of two major networks: one is a generative network (generator) and the other is a discriminative network (discriminator).
      The generator tries to generate realistic output from input, and the discriminator tries to distinguish which one is a more real-like pair between a real pair and a fake pair.  

__Generator architectures__

In our model, we use a global generator (G).
The generator is consist of the Encoder - Residual Blocks - Decoder.
The 'nd' indicate how many times you want to downsample input data, and the 'nr' indicate the number of residual blocks.

* Encoder
1. Conv2D(filter = 32, strides = 1), InstanceNorm2d, ReLU
2. Conv2D(filter = 32*2^(i_nd+1), strides = 2), InstanceNorm2d, ReLU 

* Residual Blocks (*nr)
1. Conv2D(filter = 32*2^(nd+1), strides = 1), InstanceNorm2d, ReLU
2. Conv2D(filter = 32*2^(nd+1), strides = 1), InstanceNorm2d

* Decoder
1. Conv2DTranspose(filter = 32*2^(nd+1)//2^(i_nd), strides = 2), InstanceNorm2d, ReLU
2. Conv2DTranspose(filter = 32, strides = 1)
   
__Discriminator architectures__

In our model, we use two 70*70 patch discriminator (D_1 and D_2).
One discriminator gets input pairs of the original pixel size, and the other gets input pairs which are downsampled by half.

1. Conv2D(filers = 64, strides = 2), LeakyReLu(slope = 0.2)
2. Conv2D(filers = 128, strides = 2), InstanceNorm, LeakyReLu(slope = 0.2)
3. Conv2D(filers = 256, strides = 2), InstanceNorm, LeakyReLu(slope = 0.2)
4. Conv2D(filers = 512, strides = 2), InstanceNorm, LeakyReLu(slope = 0.2)
5. Conv2D(filers = 1, strides = 1)



_____________
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
