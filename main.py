from utils.utils_cnn import *
from utils.utils_ensemble import *
import warnings
warnings.filterwarnings("ignore")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default = '.', help='Directory where the image data is stored')
parser.add_argument('--epochs', type=int, default = 20, help='Number of Epochs of training')
parser.add_argument('--batch_size', type=int, default = 4, help='Batch Size for training')
parser.add_argument('--num_workers', type=int, default = 2, help='Number of worker processes')
parser.add_argument('--learning_rate', type=float, default = 0.001, help='Learning Rate')
parser.add_argument('--momentum', type=int, default = 0.99, help='Momentum')
args = parser.parse_args()

train_resnet34(args.root,
               epochs = args.epochs,
               batch_size = args.batch_size,
               num_workers = args.num_workers,
               learning_rate = args.learning_rate,
               momentum = args.momentum,
               )

train_densenet201(args.root,
                  epochs = args.epochs,
                  batch_size = args.batch_size,
                  num_workers = args.num_workers,
                  learning_rate = args.learning_rate,
                  momentum = args.momentum,
                  )

train_inceptionv3(args.root,
                  epochs = args.epochs,
                  batch_size = args.batch_size,
                  num_workers = args.num_workers,
                  learning_rate = args.learning_rate,
                  momentum = args.momentum,
                  )

p1,labels = getfile(args.root,"csv/densenet201")
p2,_ = getfile(args.root,"csv/resnet34")
p3,_ = getfile(args.root,"csv/inception_v3")

p1 = p1[:,:-1]
p2 = p2[:,:-1]
p3 = p3[:,:-1]

# Bagging Ensemble
ensemble_prob = (p1+p2+p3)/3
prediction = predicting(ensemble_prob)

correct = np.where(prediction == labels)[0].shape[0]
total = labels.shape[0]
acc = correct/total

print("Accuracy = ",acc)
metrics(labels,prediction, classes = ['COVID', 'Non-COVID'])
