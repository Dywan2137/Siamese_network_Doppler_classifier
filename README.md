# Siamese_network_Doppler_classifier
A lightweight Siamese network + KNN classifier for identifying Doppler and Gamma Doppler knife finishes in CS2 from images.
The data for M9 Doppler|Gamma doppler is avaliable on kaggle https://www.kaggle.com/datasets/dywan2137/m9-bayonet and was scrapped from csgoskins.gg

The task is too easy for the siamese network. The accuracy is almost every time 100% which is good but the KNN does most of the work which is proven by performing KNN on a untrained model which shows high accuracy in both cases regardless of the embeddings.
The real find is when traing data is starved to only 2 images per class the trained embeddings maintain high accuracy while random embeddings have slightly lower ones. Which shows that siamese network are a viable method to use with little observations avaliable
