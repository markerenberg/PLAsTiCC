# PLAsTiCC
This repository contains code used in the PLAsTiCC Astronomical Classification Kaggle Competition. 
https://www.kaggle.com/c/PLAsTiCC-2018/overview

# Background of Competition 
The Photometric LSST Astronomical Time-Series Classification Challenge (PLAsTiCC) competition was designed to allow competitors to classify astronomical sources that vary with time into different classes. In general, it aimed to answer the question:
"How well can we classify objects in the sky from simulated LSST time-series data, with all its challenges of non-representativity?".

Competitors were given simulated LSST time-series data in the form of light curves. These light curves were the result of difference imaging, where two images are taken of the same region on different nights and then subtracted from each other. The objective of the competition was to classify these light curves into 15 classes, only 14 of which were represented in the training sample. The final class was meant to capture interesting objects that are hypothesized to exist but have never been observed.

# Feature Engineering

# Gaussian Process Regression
My first issue when trying to engineer features was to acknowledge that the recorded observations were not taken on a consistent time scale in the training data. This meant that I had to interpolate observations to create a regular time series, rather than an unevenly spaced one. To do this, I implemented a Gaussian Process Regression model to predict the mean and variance functions, to form a distribution for a Normal random variable to which the observations belong. A Gaussian Process was chosen over Spline Interpolation and Recurrent Neural Networks due to it's ability to allow for uncertainty in the interpolated space, and due to it's time efficiency. 
To implement a Gaussian Process model, the celerite package was used in Python. I used the Matern32 Kernel function as it was shown to be effective with spatial data. I then optimized the sigma and rho parameters using the limited-memory BFGS optimization algorithm. These optimized parameters were then used in the final gaussian process, which predicted the Flux variable on a consistent time scale. 

# Wavelet Transform
After much research, I decided to use the Wavelet decomposition to transform the signal data onto the time and frequency domains. 

# Our Model
The final model we used a stacked random forest model, implemented with XGBoost and LightGBM random forests. 

# Final Ranking
Our submission achieved a leaderboard score of 1.44812. We learned that the loss function used in the competition was a modified version of a weighted log loss function amongst all 15 classes. 
Our submission was ranked 628th on the leaderboard. I take this challenge as a learning opportunity, and will continue to expand on the  knowledge I have gained from this competition. I am very fortunate to have taken the time to study astronomy, photonomy, and spectroscopy, and to have been able to assemble a working model in time that accurately classified the simulated light curves.
