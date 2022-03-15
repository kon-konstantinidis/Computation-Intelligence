# Konstantinidis Konstantinos
# AEM: 9162
# email: konkonstantinidis@ece.auth.gr

from keras.datasets import mnist
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from keras import regularizers
from keras import Sequential
from keras.layers import Dense

# Load mnist dataset
(trainFeatures,trainTarget),(testFeatures,testTarget) = mnist.load_data()

## Prepare labels
# compute the number of labels
num_labels = len(np.unique(trainTarget)) # 10, I know
# convert targets to one-hot vector
trainTarget = tf.one_hot(trainTarget, depth=num_labels)
testTarget = tf.one_hot(testTarget, depth=num_labels)

## Prepare features
# image dimensions (assumed square)
image_size = trainFeatures.shape[1]
input_size = image_size * image_size
# resize and normalize features
trainFeatures = np.reshape(trainFeatures, [-1, input_size])
trainFeatures = trainFeatures.astype('float32') / 255
testFeatures = np.reshape(testFeatures, [-1, input_size])
testFeatures = testFeatures.astype('float32') / 255

iters = 100
modelChoice = 0
# 0: run all models
# 1: run default model - varying batch size
# 2: RMSprop Optimizer- varying rho
# 3: SGD Optimizer
# 4: SGD Optimizer - L2 Regularization - varying alpha
# 5: SGD Optimizer - L1 Regularization

resultsFile = open('results.txt','a')
##################################### 1. Default Model ##################################
if (modelChoice == 1 or modelChoice == 0):
    batchSizes = [1, 256, int(np.shape(trainFeatures)[0]*0.8)]
    for bS in batchSizes:
        # Create, compile and train model
        model = Sequential([
            tf.keras.layers.Input(input_size),
            Dense(128, activation='relu'),
            Dense(256, activation='relu'),
            Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
        start = time.time() # time the training process
        history = model.fit(trainFeatures, trainTarget, batch_size=bS, epochs=iters, validation_split=0.2, verbose=1)
        end = time.time()

        # Evaluate model on test set
        test_loss, test_acc = model.evaluate(testFeatures,testTarget,verbose=2)

        # Print/Write results
        resultMsg = "1. Default Model | Batch-size: {} | Duration: {} sec | Test accuracy: {} \n".format(str(bS),str(end-start),test_acc)
        print(resultMsg)
        resultsFile.write(resultMsg)
        
        ## Produce and save the required plots
        # Accuracy over training and validation data
        plt.figure()
        plt.plot(range(1,iters+1),history.history['accuracy'], label = 'train',color='red')
        plt.plot(range(1,iters+1),history.history['val_accuracy'], label = 'validation',color='green')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('1. Default Model with batch size = ' + str(bS))
        plt.savefig('1. Default Model Accuracy batch_size_' + str(bS) + '.png')
        plt.close()
        # Loss over training and validation data
        plt.figure()
        plt.plot(range(1,iters+1),history.history['loss'], label = 'train',color='red')
        plt.plot(range(1,iters+1),history.history['val_loss'], label = 'validation',color='green')
        plt.xlabel('Iteration')
        plt.ylabel('Categorical Cross-Entropy')
        plt.legend()
        plt.title('1. Default Model with batch size = ' + str(bS))
        plt.savefig('1. Default Model Loss batch_size_' + str(bS) + '.png')
        plt.close()
    resultsFile.write('\n')

##################################### 2. RMSprop Optimizer ##############################
if (modelChoice == 2 or modelChoice == 0):
    RMSProp_r = [0.01, 0.99]
    for r in RMSProp_r:
        model = Sequential([
            tf.keras.layers.Input(input_size),
            Dense(128, activation='relu'),
            Dense(256, activation='relu'),
            Dense(10, activation='softmax')
    ])
        model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=r), loss='categorical_crossentropy', metrics=['accuracy'])
        start = time.time()
        history = model.fit(trainFeatures, trainTarget, batch_size=256, epochs=iters, validation_split=0.2, verbose=1)
        end = time.time()
    
        test_loss, test_acc = model.evaluate(testFeatures,testTarget,verbose=2)    

        resultMsg = "2. RMSprop Optimizer | rho: {} | Duration: {} sec | Test accuracy: {} \n".format(str(r),str(end-start),test_acc)
        print(resultMsg)
        resultsFile.write(resultMsg)

        plt.figure()
        plt.plot(range(1,iters+1),history.history['accuracy'], label = 'train',color='red')
        plt.plot(range(1,iters+1),history.history['val_accuracy'], label = 'validation',color='green')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('2. RMSprop Optimizer with rho =' + str(r))
        plt.savefig('2. RMSprop Optimizer Accuracy rho_' + str(r) + '.png')
        plt.close()
        
        plt.figure()
        plt.plot(range(1,iters+1),history.history['loss'], label = 'train')
        plt.plot(range(1,iters+1),history.history['val_loss'], label = 'validation')
        plt.xlabel('Iteration')
        plt.ylabel('Categorical Cross-Entropy')
        plt.legend()
        plt.title('2. RMSprop Optimizer with rho =' + str(r))
        plt.savefig('2. RMSprop Optimizer Loss rho_' + str(r) + '.png')
        plt.close()
    resultsFile.write('\n')

##################################### 3. SGD Optimizer ##################################
if (modelChoice == 3 or modelChoice == 0):
    initializer = tf.keras.initializers.RandomNormal(mean=5)
    model = Sequential([
            tf.keras.layers.Input(input_size),
            Dense(128, activation='relu', kernel_initializer=initializer),
            Dense(256, activation='relu', kernel_initializer=initializer),
            Dense(10, activation='softmax', kernel_initializer=initializer)
    ])
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    start = time.time()
    history = model.fit(trainFeatures, trainTarget, batch_size=256, epochs=iters, validation_split=0.2, verbose=1)
    end = time.time()
    
    test_loss, test_acc = model.evaluate(testFeatures,testTarget,verbose=2)    
    
    resultMsg = "3. SGD Optimizer | Duration: {} sec | Test accuracy: {} \n".format(str(end-start),test_acc)
    print(resultMsg)
    resultsFile.write(resultMsg)

    plt.figure()
    plt.plot(range(1,iters+1),history.history['accuracy'], label = 'train',color='red')
    plt.plot(range(1,iters+1),history.history['val_accuracy'], label = 'validation',color='green')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('3. SGD Optimizer')
    plt.savefig('3. SGD Optimizer Accuracy.png')
    plt.close()
    
    plt.figure()
    plt.plot(range(1,iters+1),history.history['loss'], label = 'train',color='red')
    plt.plot(range(1,iters+1),history.history['val_loss'], label = 'validation',color='green')
    plt.xlabel('Iteration')
    plt.ylabel('Categorical Cross-Entropy')
    plt.legend()
    plt.title('3. SGD Optimizer')
    plt.savefig('3. SGD Optimizer Loss.png')
    plt.close()
    resultsFile.write('\n')

##################################### 4. SGD Optimizer - L2 Regularization ##############
if (modelChoice == 4 or modelChoice == 0):
    initializer = tf.keras.initializers.RandomNormal(mean=5)
    alphas = [0.1, 0.01, 0.001]
    for a in alphas:
        model = Sequential([
                tf.keras.layers.Input(input_size),
                Dense(128, activation='relu', kernel_initializer=initializer, kernel_regularizer=regularizers.l2(a)),
                Dense(256, activation='relu', kernel_initializer=initializer, kernel_regularizer=regularizers.l2(a)),
                Dense(10, activation='softmax', kernel_initializer=initializer, kernel_regularizer=regularizers.l2(a))
        ])
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
        start = time.time()
        history = model.fit(trainFeatures, trainTarget, batch_size=256, epochs=iters, validation_split=0.2, verbose=1)
        end = time.time()

        test_loss, test_acc = model.evaluate(testFeatures,testTarget,verbose=2)    

        resultMsg = "4. SGD Optimizer - L2 Regularization | a: {} | Duration: {} sec | Test accuracy: {} \n".format(str(a),str(end-start),test_acc)
        print(resultMsg)
        resultsFile.write(resultMsg)

        plt.figure()
        plt.plot(range(1,iters+1),history.history['accuracy'], label = 'train',color='red')
        plt.plot(range(1,iters+1),history.history['val_accuracy'], label = 'validation',color='green')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('4. SGD Optimizer - L2 Regularization with a =' + str(a))
        plt.savefig('4. SGD Optimizer - L2 Regularization Accuracy a_' + str(a) + '.png')
        plt.close()
        
        plt.figure()
        plt.plot(range(1,iters+1),history.history['loss'], label = 'train',color='red')
        plt.plot(range(1,iters+1),history.history['val_loss'], label = 'validation',color='green')
        plt.xlabel('Iteration')
        plt.ylabel('Categorical Cross-Entropy')
        plt.legend()
        plt.title('4. SGD Optimizer - L2 Regularization with a =' + str(a))
        plt.savefig('4. SGD Optimizer - L2 Regularization Loss a_' + str(a) + '.png')
        plt.close()
    resultsFile.write('\n')

##################################### 5. SGD Optimizer - L1 Regularization ##############
if (modelChoice == 5 or modelChoice == 0):
    model = Sequential([
        tf.keras.layers.Input(input_size),
        Dense(128, activation='relu', kernel_regularizer=regularizers.l1(0.01)),
        tf.keras.layers.Dropout(0.3),
        Dense(256, activation='relu', kernel_regularizer=regularizers.l1(0.01)),
        tf.keras.layers.Dropout(0.3),
        Dense(10, activation='softmax', kernel_regularizer=regularizers.l1(0.01))
    ])
    model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
    start = time.time()
    history = model.fit(trainFeatures, trainTarget, batch_size=256, epochs=iters, validation_split=0.2, verbose=1)
    end = time.time()

    test_loss, test_acc = model.evaluate(testFeatures,testTarget,verbose=2)  

    resultMsg = "5. SGD Optimizer - L1 Regularization | Duration: {} sec | Test accuracy: {} \n".format(str(end-start),test_acc)
    print(resultMsg)
    resultsFile.write(resultMsg)

    plt.figure()
    plt.plot(range(1,iters+1),history.history['accuracy'], label = 'train',color='red')
    plt.plot(range(1,iters+1),history.history['val_accuracy'], label = 'validation',color='green')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('5. SGD Optimizer - L1 Regularization')
    plt.savefig('5. SGD Optimizer - L1 Regularization Accuracy.png')
    plt.close()
      
    plt.figure()
    plt.plot(range(1,iters+1),history.history['loss'], label = 'train',color='red')
    plt.plot(range(1,iters+1),history.history['val_loss'], label = 'validation',color='green')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('5. SGD Optimizer - L1 Regularization')
    plt.savefig('5. SGD Optimizer - L1 Regularization Loss.png')
    plt.close()
    resultsFile.write('\n')

resultsFile.close()