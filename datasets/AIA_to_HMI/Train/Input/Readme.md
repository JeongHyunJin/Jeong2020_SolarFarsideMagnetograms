Input Data for Training Our Model
---------------------------------

* SDO/AIA three (30.4, 19.3, and 17.1 nm) passband images.
* The images have a cadence of 12 hour (at 00 UT and 12 UT each day) from 1 January 2011 to 31 December 2017.
* We construct training sets with ten months and evaluation sets with two months, and both are randomly selected for each year without any duplication between the two sets.
* Data pre-processing is applied to the EUV data for the effective training and generating.
* We make Level 1.5 images with the standard SolarSoftWare (SSW) packages of aia_prep.pro function, which process the images by calibrating, rotating and centering.
* We downsample them from 4096 X 4096 to 1024 X 1024 pixels, and the solar radius (Rsun) were fixed at 450 pixels.
* We mask the area outside 0.98 Rsun of disk center to minimize the uncertainty of limb data.
* For the calibration of all EUV data, all data numbers are scaled by median values of the original data on the solar disk, which are fixed at 100.
* Then the logarithms of the scaled data are normalized from -1 to 1 with the saturation values of 0 (lower limit) and log(200) (upper limit). 
  ( ->  pipeline.py)
* And we combine the three passband images from the SDO into the RGB channel dimensions.
* Finally we manually exclude a set of data with poor quality; for example, too noise images because of solar flares, those with incorrect header information, those with infrequent events such as eclipses, transits, etc..
