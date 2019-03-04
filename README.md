# PLAsTiCC
This repository contains code used in the PLAsTiCC Astronomical Classification Kaggle Competition. 
https://www.kaggle.com/c/PLAsTiCC-2018/overview

# Background of Competition 
The Photometric LSST Astronomical Time-Series Classification Challenge (PLAsTiCC) competition was designed to allow competitors to classify astronomical sources that vary with time into different classes. In general, it aimed to answer the question:
"How well can we classify objects in the sky from simulated LSST time-series data, with all its challenges of non-representativity?".

Competitors were given simulated LSST time-series data in the form of light curves. These light curves were the result of difference imaging, where two images are taken of the same region on different nights and then subtracted from each other. The objective of the competition was to classify these light curves into 15 classes, only 14 of which were represented in the training sample. The final class was meant to capture interesting objects that are hypothesized to exist but have never been observed.


# Our Submission
Our submission achieved a leaderboard score of 1.44812. We learned that the loss function used in the competition was a modified version of a weighted log loss function amongst all 15 classes. 
Unfortunately, this submission was only able to rank 628th on the leaderboard, however I take this challenge as a learning opportunity, and will continue to expand on the incredible knowledge I ahve gained from this competition. I am   very fortunate to have learned about astronomy, photonomy, and spectroscopy, and to have been able to assemble a working model in time that accurately classified the simulated light curves.
