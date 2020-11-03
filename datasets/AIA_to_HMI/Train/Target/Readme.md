Target Data for Training Our Model
----------------------------------

* SDO/HMI line-of-sight 720s magnetograms.
* The data have a cadence of 12 hour (at 00 UT and 12 UT each day) from 1 January 2011 to 31 December 2017.
* We construct training sets with ten months and evaluation sets with two months, and both are randomly selected for each year without any duplication between the two sets.
* Data pre-processing is applied to the magnetograms for the effective training and generating.
* We make Level 1.5 images with the standard SolarSoftWare (SSW) packages of hmi prep:pro function, which process the images by calibrating, rotating and centering.
* We downsample them from 4096 X 4096 to 1024 X 1024 pixels, and the solar radius (Rsun) were fixed at 450 pixels.
* We mask the area outside 0.98 Rsun of disk center to minimize the uncertainty of limb data.
----------------------------------
(pipeline.py)
* The magnetograms for training have the upper saturation limit of +/- 3,000 Gauss for the normalization.
* Finally we manually exclude a set of data with poor quality; for example, too noise images because of solar flares, those with incorrect header information, those with infrequent events such as eclipses, transits, etc..
