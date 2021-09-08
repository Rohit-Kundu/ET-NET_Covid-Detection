# ET-NET_Covid-Detection
Based on our paper "[ET-NET: An Ensemble of Transfer Learning Models for Prediction of COVID-19 Infection through Chest CT-scan Images](https://doi.org/10.1007/s11042-021-11319-8)" published in Springer- Multimedia Tools and Applications.

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

# Citation
If this repository helps you in any way, please consider citing our paper:
```
Kundu, Rohit, et al. "ET-NET: an ensemble of transfer learning models for prediction of COVID-19 infection through chest CT-scan images." Multimedia Tools and Applications (2021): 1-20.
```
Bibtex:
```
@article{kundu2021net,
  title={ET-NET: an ensemble of transfer learning models for prediction of COVID-19 infection through chest CT-scan images},
  author={Kundu, Rohit and Singh, Pawan Kumar and Ferrara, Massimiliano and Ahmadian, Ali and Sarkar, Ram},
  journal={Multimedia Tools and Applications},
  pages={1--20},
  year={2021},
  publisher={Springer}
}
```
