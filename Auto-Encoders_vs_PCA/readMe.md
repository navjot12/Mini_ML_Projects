# Auto-Encoders vs. PCA

This python script compares the performance of PCA with Auto-Encoders by comparing the regeneration of a dataset.

MNIST dataset (- a collection of 42000 handwritten digit (0-9) images) has been used for comparison. First half of the dataset is used for training while the subsequent quarter of data has been used for validation. Since the dataset is too heavy to be uploaded on github (76.8 MB), it can be found at https://goo.gl/Wyl4hX.

#### Auto Encoder

The script uses Keras based on Theano backend generate an auto encoder comprising of-  
	1. Input layer accepting digits of MNIST dataset, having shape (784,).  
	2. **ENCODER** : First hidden layer: Embedding input of shape (784,) into shape (64,).  
	3. **DECODER** : Output layer: Regenerating an output (a digit) of shape (784,) from an input of shape of (64,).  

The first 2 layers are mapped to an encoder while the last two layers are mapped to a decoder.

#### Principal Component Analysis

The script uses PCA from sklearn.decomposition to fit and transform 784-dimension dataset to 64-dimension data. The 64-dimension data is inverse-transformed to achieve back a 784-dimension result- representing the regenerated digit.

### Result Summary:
![alt text](https://github.com/navjot12/Mini_ML_Projects/blob/master/Auto-Encoders_vs_PCA/Result.png "Result")

**The Auto-Encoder visibly provides much better results than PCA.**

However, the Auto-Encoder took approximately 3 minutes to run 50 epochs while PCA provided almost instantaneous results.
