# ET-NET_Covid-Detection
Based on our paper "ET-NET: An Ensemble of Transfer Learning Models for Prediction of COVID-19 Infection through Chest CT-scan Images" accepted for publication in Springer- Multimedia Tools and Applications.

# Required Dependencies

To install the dependencies, run the following using the command prompt:

`pip install -r requirements.txt`

## Running the code on the COVID data
Download the dataset from [Kaggle](https://www.kaggle.com/plameneduardo/sarscov2-ctscan-dataset) and split it into 5-fold cross-validation train and validation sets.

Required Directory Structure:
```

+-- data
|   +-- .
|   +-- train
|   +-- val
+-- utils
|   +-- .
|   +-- utils_cnn
|   +-- utils_ensemble.py
+-- main.py

```
To run the ensemble model on the base learners run the following:

`python main.py --root "path/"`

Available arguments:
- `--epochs`: Number of epochs of training. Default = 100
- `--batch_size`: Batch Size. Default = 4
- `--num_workers`: Number of Worker processes. Default = 2
- `--learning_rate`: Learning Rate. Default = 0.001
- `--momentum`: Momentum value. Default = 0.99
