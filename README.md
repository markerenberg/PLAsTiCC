# PLAsTiCC Astronomical CLassification Kaggle
This repository contains code used in the PLAsTiCC Astronomical Classification Kaggle Competition. 
https://www.kaggle.com/c/PLAsTiCC-2018/overview

# Background of Competition 
The Photometric LSST Astronomical Time-Series Classification Challenge (PLAsTiCC) competition was designed to allow competitors to classify astronomical sources that vary with time into different classes. In general, it aimed to answer the question:
"How well can we classify objects in the sky from simulated LSST time-series data, with all its challenges of non-representativity?".

Competitors were given simulated LSST time-series data in the form of light curves. These light curves were the result of difference imaging, where two images are taken of the same region on different nights and then subtracted from each other. The objective of the competition was to classify these light curves into 15 classes, only 14 of which were represented in the training sample. The final class was meant to capture interesting objects that are hypothesized to exist but have never been observed.

# Our Pipeline

## Train Test Split
The training data was split using a stratified K-fold split, with a K-value of 10. This method was proven to be the most effective after comparing with other alternatives, such as grouping observations by class and performing random stratified sampling amongst the classes.

## Feature Engineering
The first set of features engineered were simple sums and measures of central tendency related to the flux and mjd variables (X and Y respectively). In addition to this, the tsfresh library was used to extract features related to periodicity and distribution of the light curves. 

After extracting these features, an issue I faced was that that the recorded observations were not taken on a consistent time scale in the training data. This meant that I had to interpolate observations to create a regular time series, rather than an unevenly spaced one. To do this, I implemented a Gaussian Process Regression Encoder to create a fixed-length series of observations from the irregularly timed input. A Gaussian Process was chosen over Spline Interpolation and Recurrent Neural Networks due to it's ability to allow for uncertainty in the interpolated space, and due to it's run-time efficiency. 
To implement a Gaussian Process model, the celerite package was used in Python. The model was implemented using the Matern32 Kernel function, as it was shown to be more effective with spatial data.

After obtaining a consistent time series of data, the Wavelet transform algorithm was applied to extract time and frequency information from the original signal, while reducing photometric noise from the data set. The reason the Stationary Wavelet Transform (SWT) was used instead of the Discrete Wavelet Transform (DWT) is because the DWT is not time-invariant, and is therefore sensitive to the alignment of the light curves with time.  Once the approximation and detail coefficients had been extracted, Principal Component Analysis (PCA) was run to reduce the dimensions of the input.

## Our Model
The final model we used a stacked random forest model, implemented with XGBoost and LightGBM random forests. 

## Evaluation
Thanks to assistance from kernels and discussions, it was discovered that the leaderboard used a modified version of a multi-weighted log loss algorithm. As such, this loss function was used to evaluate the predictive accuracy of our final model. 

# Final Ranking
Our submission achieved a leaderboard score of 1.44812. We learned that the loss function used in the competition was a modified version of a weighted log loss function amongst all 15 classes. 
Our submission was ranked 628th on the leaderboard. I take this challenge as a learning opportunity, and will continue to expand on the  knowledge I have gained from this competition. I am very fortunate to have taken the time to study astronomy, photonomy, and spectroscopy, and to have been able to assemble a working model in time that accurately classified the simulated light curves.
