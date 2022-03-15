# Konstantinidis Konstantinos
# AEM: 9162
# email: konkonstantinidis@ece.auth.gr

import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from rbflayer import RBFLayer
from keras.layers.core import Dense
from keras.models import Sequential
from kmeans_initializer import InitCentersKMeans
from sklearn.model_selection import train_test_split
from keras import backend as K
import matplotlib.pyplot as plt

# Function for calculating R2 metric via tensors
def R2(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res/(SS_tot + K.epsilon())

# Load Boston Housing dataset and split it to 75% train set and 25% test set
(trainFeatures,trainTarget),(testFeatures,testTarget) = tf.keras.datasets.boston_housing.load_data(test_split=0.25,seed=21)

# Split train data to 80% train and 20% validation
(trainFeatures,valFeatures,trainTarget,valTarget) = train_test_split(trainFeatures,trainTarget,test_size=0.2)

# Standardize data to 0 mean and unit variance
scaler = StandardScaler()
testFeatures = scaler.fit_transform(testFeatures)
valFeatures = scaler.fit_transform(valFeatures)
trainFeatures = scaler.fit_transform(trainFeatures)

# For the implementation of an RBF layer, we will be using: https://github.com/PetraVidnerova/rbf_keras (with some slight fixes)
# Create the RBF-NN for the different parameters
neuronsRatio = [0.1, 0.5, 0.9]
trainR2s = list()
trainRMSEs = list()
valR2s = list()
valRMSEs = list()
testR2s = list()
testRMSEs = list()
for i in range(0,3):
    model = Sequential()
    numNeurons = int(neuronsRatio[i]*np.shape(trainFeatures)[0])
    rbflayer = RBFLayer(numNeurons, initializer=InitCentersKMeans(trainFeatures), betas=1,input_shape = (np.shape(trainFeatures)[1],))
    model.add(rbflayer) # first layer can also be the rbf layer, no need to have an input layer
    model.add(Dense(128)) # this does significantly improve the perfomance, however
    model.add(Dense(1)) # we need one output at the last layer, not 128

    model.compile(loss='mean_squared_error',optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),metrics=[tf.keras.metrics.RootMeanSquaredError(),R2])

    history = model.fit(trainFeatures,trainTarget,batch_size=32,epochs=100,verbose=0,validation_data=(valFeatures,valTarget))

    # Extract metrics
    train_R2 = history.history['R2']
    trainR2s.append(train_R2)
    train_RMSE = history.history['root_mean_squared_error']
    trainRMSEs.append(train_RMSE)
    val_R2 = history.history['val_R2']
    valR2s.append(val_R2)
    val_RMSE = history.history['val_root_mean_squared_error']
    valRMSEs.append(val_RMSE)

    # Plot model's learning curve for training and validation data
    plt.figure()
    plt.plot(np.asarray(train_RMSE),color='red',label='Training RMSE')
    plt.plot(np.asarray(val_RMSE),color='green',label='Validation RMSE')
    plt.xlabel('Iteration')
    plt.ylabel('RMSE')
    plt.title('Learning curve of model '+ str(i+1))
    plt.legend(loc="upper right")

    (test_MSE,test_RMSE,test_R2) = model.evaluate(testFeatures,testTarget,verbose=0)
    testR2s.append(test_R2)
    testRMSEs.append(test_RMSE)

# Print the metrics
for i in range(0,3):
    print('Model with neurons ratio=',neuronsRatio[i],' metrics:',sep='')
    print('RMSE:',round(testRMSEs[i],3),'    R2:',round(testR2s[i],3))

# Show plots
plt.show()