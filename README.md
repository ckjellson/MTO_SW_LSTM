# Many-to-one sliding window LSTM in Pytorch

* Many-to-one LSTM using sliding window for arbitrary and varying sequence lengths.
* Can be set to use GPU.
* Uses zero-padding to get an equal number of windows fitted to the sequence lengths using the chosen stride.

Model was the basis for the main contribution to estimation of relative positions in 2nd place submission to Kaggle-competition https://www.kaggle.com/c/indoor-location-navigation/overview

# Files

* **MTO_SW_LSTM.py** - Model class.
* **test_MTO_SW_LSTM.ipynb** - Simple demonstration of how to train the MTO_SW_LSTM.
