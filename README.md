#
To be able to run the code, first download the training, val, and test data from ISIC 2017
Data can be downloaded from https://challenge.kitware.com/#challenge/583f126bcad3a51cc66c8d9a ,use only part 1:segmentation

Create a folder called data. In data, create 3 folders and name it test,val,train. In each of the test,val,train folders, create 2 additional folders(one for the ground truth, the other for the input images).

For the ground_truth folder, name it:ISIC-2017_type_GroundTruth, for the input images folder name it: ISIC-2017_type_Data,where type is either "Training","Test", or "Validation".

lesion.py is used to load up the data

eval_probabilistic.ipynb is used to calculate accuracy, jaccard ind, and ece

ensemble_ent.ipynb is used to calculate pac,pui of either  deterministic and mc ensembles

ensemble_ece.ipynb is used to calculate ece,accuracy, and jaccard ind of ensembles

entropy_threshold.ipynb calculates pac and pui for  models using entropy --> only for bayesian models

threshold.ipynb calculates pac and pui using the confidence or variance of MC samples

visulaization.ipynb plots the output

residual_unet.py has several unet variants including gated res unet and residual unet using add or concat. Gaussian Dropout is also in here.

ECE.py is taken from Gpleiss

