# Konstantinidis Konstantinos
# AEM: 9162
# email: konkonstantinidis@ece.auth.gr

from keras.datasets import mnist
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import regularizers
from keras import Sequential
from keras.layers import Dense
from keras.initializers.initializers_v2 import HeNormal
from keras import backend as K
from keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
from alive_progress import alive_bar
from sklearn.metrics import confusion_matrix

def recall(y_true, y_pred):
    #y_true = K.ones_like(y_true) 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    
    recall = true_positives / (all_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    #y_true = K.ones_like(y_true) 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_score(y_true, y_pred):
    precisionScore = precision(y_true, y_pred)
    recallScore = recall(y_true, y_pred)
    return 2*((precisionScore*recallScore)/(precisionScore+recallScore+K.epsilon()))

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
testFeatures = testFeatures.astype('float32') / 2551111

# Create hyperparameter space
nh1 = [64,128]
nh2 = [256,512]
alpha = [0.1, 0.001, 0.000001]
lr = [0.1, 0.01, 0.001]

p1Len=len(nh1)
p2Len=len(nh2)
p3Len=len(alpha)
p4Len=len(lr)

with alive_bar(p1Len*p2Len*p3Len*p4Len) as bar: # Use a cool progress bar as well
    f1_scores = np.zeros(shape=(p1Len,p2Len,p3Len,p4Len)) #f1_scores over the parameter space
    # Begin the grid search
    for p1 in range(0,len(nh1)):
        for p2 in range(0,len(nh2)):
            for p3 in range(0,len(alpha)):
                for p4 in range(0,len(lr)):
                    # Create the model
                    model = Sequential([
                        tf.keras.layers.Input(input_size),
                        Dense(units=nh1[p1], activation='relu',kernel_regularizer=regularizers.l2(alpha[p3]),kernel_initializer=HeNormal()),
                        Dense(units=nh2[p2], activation='relu',kernel_regularizer=regularizers.l2(alpha[p3]),kernel_initializer=HeNormal()),
                        Dense(10, activation='softmax',kernel_regularizer=regularizers.l2(alpha[p3]),kernel_initializer=HeNormal())
                    ])
                    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=lr[p4]), loss='categorical_crossentropy', metrics=f1_score)
                    # Begin k-fold cross validation of the model
                    cvf1s = list()
                    kf = KFold(n_splits=5)
                    print('CV starting...')
                    k=1
                    for train_index,val_index in kf.split(trainFeatures,trainTarget):
                        trainFeaturesCV = trainFeatures[train_index,]
                        valFeaturesCV = trainFeatures[val_index,]
                        trainTargetCV = tf.gather(trainTarget,train_index)
                        valTargetCV = tf.gather(trainTarget,val_index)
                        history = model.fit(trainFeatures, trainTarget, batch_size=512, epochs=1000, verbose=0, callbacks=EarlyStopping(monitor = 'f1_score', patience=200))
                        (cv_CCE,cv_f1) = model.evaluate(valFeaturesCV,valTargetCV,verbose=0)
                        cvf1s.append(cv_f1)
                        print('Fold ',k,' | f1_score: ',cv_f1,sep='')
                        k = k+1
                    # k-fold cross validation has ended, get the mean f1_score for this parameter combo
                    f1_scores[p1,p2,p3,p4] = np.nanmean(cvf1s)
                    print('CV ended, f1_score: ',np.nanmean(cvf1s))
                    # Make progress bar go forward
                    bar()
# Find best parameter combo (highest F-measure)
maxF1_score = np.amax(f1_scores)
maxIndexes = np.where(f1_scores == maxF1_score)
best_nh1 = nh1[maxIndexes[0][0]]
best_nh2 = nh2[maxIndexes[1][0]]
best_alpha = alpha[maxIndexes[2][0]]
best_lr = lr[maxIndexes[3][0]]
msg = "Best model has F-Measure = {} with parameters:\n{} neurons on 1st hidden layer\n{} neurons on 2nd hideen layer\n{} regularization parameter\n{} learning rate".format(maxF1_score,best_nh1,best_nh2,best_alpha,best_lr)
print(msg)
resultsFile = open('results2.txt','a')
resultsFile.write(msg)

"""
# Best parameters
best_nh1=64
best_nh2=512
best_alpha = 0.000001
best_lr = 0.01 """

# Make the model with the optimal parameters
model = Sequential([
    tf.keras.layers.Input(input_size),
    Dense(units=best_nh1, activation='relu',kernel_regularizer=regularizers.l2(best_alpha),kernel_initializer=HeNormal()),
    Dense(units=best_nh2, activation='relu',kernel_regularizer=regularizers.l2(best_alpha),kernel_initializer=HeNormal()),
    Dense(10, activation='softmax',kernel_regularizer=regularizers.l2(best_alpha),kernel_initializer=HeNormal())
])
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=best_lr), loss='categorical_crossentropy', metrics=['accuracy',recall, precision, f1_score])
history = model.fit(trainFeatures, trainTarget, batch_size=512, epochs=1000, verbose=0, callbacks=EarlyStopping(monitor = 'f1_score', patience=200),validation_split=0.2)

# Extract metrics
best_train_fMeasure = history.history['f1_score']
best_val_fMeasure = history.history['val_f1_score']

# Plot best model's learning curve for training and validation data
plt.figure()
plt.plot(np.asarray(best_train_fMeasure),color='red',label='Training F-Measure')
plt.plot(np.asarray(best_val_fMeasure),color='green',label='Validation F-Measure')
plt.xlabel('Iteration')
plt.ylabel('F-Measure')
plt.title('Learning Curve of Best Model')
plt.legend(loc="upper right")
plt.savefig('Learning Curve - Best Model.png')
plt.close()

(bestModel_CCE,bestModel_acc,bestModel_recall,bestModel_precision,bestModel_fMeasure) = model.evaluate(testFeatures,testTarget)

msg = "| Accuracy: {}\n| Recall: {}\n| Precision: {}\n| F-Measure: {}\n".format(round(bestModel_acc,3),round(bestModel_recall,3),round(bestModel_precision,3),round(bestModel_fMeasure,3))
print('Best parameterized model on test set:')
print(msg)

resultsFile.write('Best parameterized model on test set:\n')
resultsFile.write(msg)

# Helper function for plotting nicely a classification model's confusion matrix
def plot_confusion_matrix(cm,target_names, title='Confusion matrix', cmap=None, normalize=False):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import itertools
    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(title+'.png')
    plt.close()

# Get confusion matrix
predictions = model.predict(testFeatures)
predictions = np.argmax(predictions, axis=1) # get indices of max values per row (essentialy transform back from one-hot)
testTarget = np.argmax(testTarget,axis=1) # same for testTarget
labels = ['0','1','2','3,','4','5','6','7','8','9']
plot_confusion_matrix(confusion_matrix(testTarget,predictions),target_names=labels)