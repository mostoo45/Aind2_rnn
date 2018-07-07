import numpy as np

from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    #print(series.shape)
    for i in range(0,series.shape[0]-window_size):
        #print(i,i+window_size)
        #print(i,i+window_size,window_size+i+1)
        X.append(series[i:i+window_size])
        y.append(series[window_size+i])
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    #print(X)
    #print(y)
    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    #pass
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size,1),dropout=0.1))
    #model.add(Dropout(0.1))
    #model.add(Dense(15)) # (timestep, feature)
    #model.add(Dropout(0.1))
    model.add(Dense(1)) # output = 1 
    #model.compile(loss='mean_squared_error', optimizer='adam') 
    #model.summary()
    return model

### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']

    chars = sorted(list(set(text)))
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    text = text.replace('-', ' ')
    text = text.replace('*', ' ')
    text = text.replace('/', ' ')
    text = text.replace('&', ' ')
    text = text.replace('%', ' ')
    text = text.replace('@', ' ')
    text = text.replace('$', ' ')
    text = text.replace('à', ' ')
    text = text.replace('â', ' ')
    text = text.replace('è', ' ')
    text = text.replace('é', ' ')
    text = text.replace('(', ' ')
    text = text.replace(')', ' ')
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    
    print(len(text))
    print(len(text)-step_size)
    for i in range(0,len(text)-window_size,step_size):
        #print(i,i+window_size)
        #print(i)
        inputs.append(text[i:i+window_size])
        outputs.append(text[window_size+i])
        
    #print(text)
    #print(window_size)
    #print(step_size)
    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    print(num_chars)
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size,num_chars),dropout=0.1)) 
    model.add(Dense(output_dim=num_chars, activation='softmax'))
    return model
