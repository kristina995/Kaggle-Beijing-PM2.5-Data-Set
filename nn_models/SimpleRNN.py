import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Dropout,SimpleRNN

win_length = 100
batch_size = 128
patience = 2
train_epochs = 50
full_epochs = 15
random_state = 123
test_size = 0.10
n_future = 100
zoom_days = 16
cutoff_number = 100
shuffle = False
optimizer = 'adam'
loss = 'mse'
early_stop_mode = 'min'
name = 'SimpleRNN'

def create_model(data):
    model = Sequential()
    model.add(SimpleRNN(128,activation='relu',input_shape=(win_length,data.shape[1]),return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Dense(data.shape[1]))

    return model