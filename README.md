# Evaluation of 3-version Machine Learning system with MNIST

The goal of this project is to compare the reliability of different 3-version machine learning (ML) models able to classify images of hand-written digits.
After developing three different models which will be :
- k-nearest neighbors (kNN)
- Convolutional Neural Network (CNN)
- Random Forest
  
We will first build a TMSI (Triple Model Single Input) system that will train its three models with the raw MNIST images without any preprocessing.

We will then build a TMTI (Triple Model Triple Input) system that will train its three models with, this time, different inputs. So, we will use three different datasets :
- The initial images
- Images with noise
- Rotated images
and assign one dataset for each model.
  
The final step will be to compute the reliability of the two systems and compare them.