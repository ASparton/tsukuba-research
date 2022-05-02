# Research on 3-version Machine Learning systems

The goal of this project is to compare the reliability of different 3-version machine learning (ML) models able to classify images.

## Part 1 : MNIST dataset
After developing three different models which will be :
- k-nearest neighbors (kNN)
- Convolutional Neural Network (CNN)
- Random Forest
  
We first build a TMSI (Triple Model Single Input) system that will train its three models with the raw MNIST images without any preprocessing.

Then we build the three possible SMTI (Single Model Triple Inputs) systems.

And we finally build a TMTI (Triple Model Triple Input) system that will train its three models with, this time, different inputs. So, we will use three different datasets :
- The initial images
- Images with noise
- Rotated images
and assign one dataset for each model.
  
The final step will be to compute the reliability of the three systems and compare them.

## Part 2 : Arabic traffic signs dataset
