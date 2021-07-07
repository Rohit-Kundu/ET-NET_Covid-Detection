import pandas as pd
import numpy as np
import os
from sklearn.metrics import *

def getfile(root, filename):
    if root[-1]!='/':
        root+='/'
    if '.csv' not in filename:
        filename = filename+'.csv'
    file = root+filename
    df = pd.read_csv(file,header=None)
    df = np.asarray(df)

    labels=[]
    classes = os.listdir(root+'val/')
    for en,c in enumerate(classes):
        for i in range(len(os.listdir(root+'val/'+c))):
            labels.append(en)
    labels = np.asarray(labels)
    return df,labels

def predicting(ensemble_prob):
    prediction = np.zeros((ensemble_prob.shape[0],))
    for i in range(ensemble_prob.shape[0]):
        temp = ensemble_prob[i]
        t = np.where(temp == np.max(temp))[0][0]
        prediction[i] = t
    return prediction

def metrics(labels,predictions,classes):
    print("Classification Report:")
    print(classification_report(labels, predictions, target_names = classes,digits = 4))
    matrix = confusion_matrix(labels, predictions)
    print("Confusion matrix:")
    print(matrix)
    print("\nClasswise Accuracy :{}".format(matrix.diagonal()/matrix.sum(axis = 1)))
    print("\nBalanced Accuracy Score: ",balanced_accuracy_score(labels,predictions))
