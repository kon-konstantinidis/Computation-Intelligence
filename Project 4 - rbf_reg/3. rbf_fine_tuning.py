# Konstantinidis Konstantinos
# AEM: 9162
# email: konkonstantinidis@ece.auth.gr

import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from rbflayer import RBFLayer
from keras.layers.core import Dense
from keras.models import Sequential
from keras.layers import Dropout
from kmeans_initializer import InitCentersKMeans
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from alive_progress import alive_bar
from sklearn.model_selection import train_test_split
from keras import backend as K

# Load Boston Housing dataset and split it to 75% train set and 25% test set
(trainFeatures,trainTarget),(testFeatures,testTarget) = tf.keras.datasets.boston_housing.load_data(test_split=0.25,seed=21)

# Standardize data to 0 mean and unit variance
scaler = StandardScaler()
testFeatures = scaler.fit_transform(testFeatures)
trainFeatures = scaler.fit_transform(trainFeatures)

# Set up the parameter grid
neuronsRBF = [0.05, 0.15, 0.3, 0.5]
neuronsRBF = [int(element * (np.shape(trainFeatures)[0])) for element in neuronsRBF]
neurons2ndLayer = [32, 64, 128, 256]
dropoutProb = [0.2, 0.35, 0.5]
p1Len=len(neuronsRBF)
p2Len=len(neurons2ndLayer)
p3Len=len(dropoutProb)

# Make a cool progress bar as well
with alive_bar(p1Len*p2Len*p3Len) as bar:
    RMSEs = np.zeros(shape=(p1Len,p2Len,p3Len))
    for p1 in range(0,p1Len):
        for p2 in range(0,p2Len):
            for p3 in range(0,p3Len):
                # Parameters
                neurons1stLayer = neuronsRBF[p1]
                neuronsDense = neurons2ndLayer[p2]
                p = dropoutProb[p3]
                # Create model
                model = Sequential()
                rbflayer = RBFLayer(neurons1stLayer,initializer=InitCentersKMeans(trainFeatures), betas=1.5,input_shape = (np.shape(trainFeatures)[1],))
                model.add(rbflayer) # first layer can also be the rbf layer, no need to have an input layer
                model.add(Dense(neuronsDense))
                model.add(Dropout(p))
                model.add(Dense(1))
                # Compile model
                model.compile(loss='mean_squared_error',optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),metrics=tf.keras.metrics.RootMeanSquaredError())
                # Begin k-fold cross validation of the model
                cvRMSEs = list()
                kf = KFold(n_splits=5)
                for train_index,val_index in kf.split(trainFeatures,trainTarget):
                    trainFeaturesCV = trainFeatures[train_index,]
                    trainTargetCV = trainTarget[train_index,]
                    valFeaturesCV = trainFeatures[val_index,]
                    valTargetCV = trainTarget[val_index,]
                    history = model.fit(trainFeaturesCV,trainTargetCV,batch_size=32,epochs=100,verbose=0)
                    (cvMSE,cvRMSE) = model.evaluate(valFeaturesCV,valTargetCV,verbose=0)
                    cvRMSEs.append(cvRMSE)
                # k-fold cross validation has ended, get the mean RMSE for this parameter combo
                RMSEs[p1,p2,p3] = np.nanmean(cvRMSEs)
                # Make progress bar go forward
                bar()

# Plot the RMSE relative to the number of RBF and 2nd layer neurons and the dropout probability
# In MATLAB, because sleep is important too, so save the data and do that there
np.savetxt('RMSE_p_0_v1.txt',RMSEs[:,:,0])
np.savetxt('RMSE_p_1_v1txt',RMSEs[:,:,1])
np.savetxt('RMSE_p_2_v1.txt',RMSEs[:,:,2])

# Find best parameter combo (lowest RMSE)
minIndexes = np.where(RMSEs == np.amin(RMSEs))
bestNeuronsRBF = neuronsRBF[minIndexes[0][0]]
bestNeurons2ndLayer = neurons2ndLayer[minIndexes[1][0]]
bestDropoutProb = dropoutProb[minIndexes[2][0]]
print('Best model has ',bestNeuronsRBF,' neurons on RBF layer, ',bestNeurons2ndLayer,' neurons on 2nd layer and ',bestDropoutProb,' dropout probablility.',sep='')
# Build optimal model

# Function for calculating R2 metric via tensors
def R2(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res/(SS_tot + K.epsilon())

# Split train data to 80% train and 20% validation
(trainFeatures,valFeatures,trainTarget,valTarget) = train_test_split(trainFeatures,trainTarget,test_size=0.2)
model = Sequential()
rbflayer = RBFLayer(bestNeuronsRBF,initializer=InitCentersKMeans(trainFeatures), betas=1.5,input_shape = (np.shape(trainFeatures)[1],))
model.add(rbflayer)
model.add(Dense(bestNeurons2ndLayer))
model.add(Dropout(bestDropoutProb))
model.add(Dense(1))
# Compile model
model.compile(loss='mean_squared_error',optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),metrics=[tf.keras.metrics.RootMeanSquaredError(),R2])
# Fit model
history = model.fit(trainFeatures,trainTarget,batch_size=32,epochs=100,verbose=0,validation_data=(valFeatures,valTarget))

# Extract metrics
best_train_RMSE = history.history['root_mean_squared_error']
best_val_RMSE = history.history['val_root_mean_squared_error']

# Plot best model's learning curve for training and validation data
plt.figure()
plt.plot(np.asarray(best_train_RMSE),color='red',label='Training RMSE')
plt.plot(np.asarray(best_val_RMSE),color='green',label='Validation RMSE')
plt.xlabel('Iteration')
plt.ylabel('RMSE')
plt.title('Learning Curve of Best Model')
plt.legend(loc="upper right")

(bestModelMSE,bestModelRMSE,bestModelR2) = model.evaluate(testFeatures,testTarget)
print('Best parameterized model on test set:')
print('MSE: ',round(bestModelMSE,3),'    RMSE: ',round(bestModelRMSE,3),'   R2: ',round(bestModelR2,3))

plt.show()