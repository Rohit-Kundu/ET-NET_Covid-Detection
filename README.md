# ET-NET_Covid-Detection
Based on our paper "ET-NET: An Ensemble of Transfer Learning Models for Prediction of COVID-19 Infection through Chest CT-scan Images" under review in Springer- Multimedia Tools and Applications.

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
+-- sar-cov-2_csv
|   +-- .
|   +-- densenet201.csv
|   +-- inception_v3.csv
|   +-- resnet34.csv
+-- main.py
+-- probability_extraction
+-- utils_ensemble.py

```
To extract the probabilities on the validation set using the different models run `probability_extraction.py` and save the files in a folder. As an example the probabilities extracted on the SARS-COV-2 dataset has been saved in the folder named `sars-cov-2_csv/`.

Next, to run the ensemble model on the base learners run the following:

`python main.py --data_directory "sars-cov-2_csv/"`
