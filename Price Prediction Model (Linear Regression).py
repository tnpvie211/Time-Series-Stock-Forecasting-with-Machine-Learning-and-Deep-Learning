#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import datetime as dt
#plot
import matplotlib.pyplot as plt
import math
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


get_ipython().system('pip install yahooquery')


# In[3]:


import yahooquery as yq


# In[4]:


def get_historical_data(company, start_date, end_date):
    ''' Date format='yyyy-mm-dd' '''
    company = company.upper()
    start = dt.date(int(start_date[0:4]), int(start_date[5:7]), int(start_date[8:10]))
    end = dt.date(int(end_date[0:4]), int(end_date[5:7]), int(end_date[8:10]))
    
    ticker = yq.Ticker(company, asynchronous= True)
    df = ticker.history(start= start, end= end).reset_index()
    col_names = ['date','open','high','low','close','volume']
    df = df[col_names]
    
    return df


# In[5]:


df = get_historical_data('GOOGL','2005-01-01','2017-06-30') # from January 1, 2005 to June 30, 2017
df.head()


# In[6]:


df.describe()


# In[7]:


df.info()


# In[8]:


from sklearn.preprocessing import MinMaxScaler

def remove_irrelevant_data(df):
    columns =['open', 'close', 'volume']
    df= df[columns].reset_index()
    return df

def normalized_data(df):
    scaler = MinMaxScaler()
    numerical = ['open', 'close', 'volume']
    df[numerical] = scaler.fit_transform(df[numerical])
    return df


# In[9]:


df = remove_irrelevant_data(df)
df


# In[10]:


def plot_company_stocks(stocks):

    fig, ax = plt.subplots()
    ax.plot(stocks['index'], stocks['close'])

    ax.set_title('Company Stock')
    plt.xlabel('Trading Days')
    plt.ylabel('Price USD')

    plt.show()


# In[11]:


df = normalized_data(df)
df


# In[12]:


plot_company_stocks(df)


# In[13]:


#Save preprocessed stock
df.to_csv('stock_preprocessed.csv',index= False)


# In[14]:


def scale_range(x, input_range, target_range):
    range = [np.amin(x), np.amax(x)]
    x_std = (x - input_range[0]) / (1.0*(input_range[1] - input_range[0]))
    x_scaled = x_std * (1.0*(target_range[1] - target_range[0])) + target_range[0]
    return x_scaled, range


# In[15]:


def train_test_split_lr(df):
    feature = []
    label = []

    # Convert dataframe columns to numpy arrays for scikit learn
    for index, row in df.iterrows():
        # print([np.array(row['Item'])])
        feature.append([(row['index'])])
        label.append([(row['close'])])

    # Regularize the feature and target arrays and store min/max of input data for rescaling later
    feature_bounds = [min(feature), max(feature)]
    feature_bounds = [feature_bounds[0][0], feature_bounds[1][0]]
    label_bounds = [min(label), max(label)]
    label_bounds = [label_bounds[0][0], label_bounds[1][0]]

    feature_scaled, feature_range = scale_range(np.array(feature), input_range=feature_bounds, target_range=[-1.0, 1.0])
    label_scaled, label_range = scale_range(np.array(label), input_range=label_bounds, target_range=[-1.0, 1.0])

    # Define Test/Train Split 80/20
    split = .315
    split = int(math.floor(len(df['index']) * split))

    # Set up training and test sets
    X_train = feature_scaled[:-split]
    X_test = feature_scaled[-split:]

    y_train = label_scaled[:-split]
    y_test = label_scaled[-split:]

    return X_train, X_test, y_train, y_test, label_range


# In[16]:


X_train, X_test, y_train, y_test, label_range= train_test_split_lr(df)

print("x_train", X_train.shape)
print("y_train", y_train.shape)
print("x_test", X_test.shape)
print("y_test", y_test.shape)


# In[17]:


from sklearn import linear_model

def build_model_linear_regression(X, y):
    linear_mod = linear_model.LinearRegression()
    X = np.reshape(X, (X.shape[0], 1))
    y = np.reshape(y, (y.shape[0], 1))
    linear_mod.fit(X, y)  # fitting the data points in the model

    return linear_mod


# In[18]:


model = build_model_linear_regression(X_train,y_train)


# In[19]:


def scale_range(x, input_range, target_range):

    range = [np.amin(x), np.amax(x)]
    x_std = (x - input_range[0]) / (1.0*(input_range[1] - input_range[0]))
    x_scaled = x_std * (1.0*(target_range[1] - target_range[0])) + target_range[0]
    return x_scaled, range


# In[20]:


def predict_prices(model, x, label_range):
    
    x = np.reshape(x, (x.shape[0], 1))
    predicted_price = model.predict(x)
    predictions_rescaled, re_range = scale_range(predicted_price, input_range=[-1.0, 1.0], target_range=label_range)

    return predictions_rescaled.flatten()


# In[21]:


predictions = predict_prices(model,X_test, label_range)


# In[22]:


fig = plt.figure()
ax = fig.add_subplot(111)

# Add labels
plt.ylabel('Price USD')
plt.xlabel('Trading Days')

# Plot actual and predicted close values
plt.plot(y_test, '#00FF00', label='Actual')
plt.plot(predictions, '#0000FF', label='Prediction')

# Set title
ax.set_title('Google Trading vs Prediction')
ax.legend(loc='upper left')

plt.show()


# In[23]:


from sklearn.metrics import mean_squared_error

trainScore = mean_squared_error(X_train, y_train)
print('Train Score: %.4f MSE (%.4f RMSE)' % (trainScore, math.sqrt(trainScore)))

testScore = mean_squared_error(predictions, y_test)
print('Test Score: %.8f MSE (%.8f RMSE)' % (testScore, math.sqrt(testScore)))


# # Long-Sort Term Memory Model

# In[32]:


def train_test_split_lstm(stocks, prediction_time=1, test_data_size=450, unroll_length=50):
    # training data
    test_data_cut = test_data_size + unroll_length + 1

    x_train = stocks[0:-prediction_time - test_data_cut].values
    y_train = stocks[prediction_time:-test_data_cut]['close'].values

    # test data
    x_test = stocks[0 - test_data_cut:-prediction_time].values
    y_test = stocks[prediction_time - test_data_cut:]['close'].values

    return x_train, x_test, y_train, y_test


# In[33]:


def unroll(data, s_length=24):
    result = []
    for ind in range(len(data)-s_length):
        result.append(data[ind: ind+s_length])
    return np.asarray(result)


# In[34]:


stocks = pd.read_csv('stock_preprocessed.csv')
stocks_data = stocks.drop(['index'], axis =1)
stocks_data.head()


# In[35]:


X_train, X_test,y_train, y_test = train_test_split_lstm(stocks_data, 5)

X_train = unroll(X_train)
X_test = unroll(X_test)
y_train = y_train[-X_train.shape[0]:]
y_test = y_test[-X_test.shape[0]:]

print("x_train", X_train.shape)
print("y_train", y_train.shape)
print("x_test", X_test.shape)
print("y_test", y_test.shape)


# In[38]:


pip install keras


# In[40]:


pip install tensorflow


# In[48]:


from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
import time


# In[50]:


def build_lstm_model(input_dim, output_dim, return_sequences):
    model = Sequential()
    model.add(LSTM(
        input_shape=(None, input_dim),
        units=output_dim,
        return_sequences=return_sequences))

    model.add(LSTM(
        100,
        return_sequences=False))

    model.add(Dense(
        units=1))
    model.add(Activation('linear'))

    return model


# In[53]:


# build basic lstm model
model = build_lstm_model(input_dim = X_train.shape[-1],output_dim = 50, return_sequences=True)

# Compile the model
start = time.time()
model.compile(loss='mean_squared_error', optimizer='adam')
print('compilation time : ', time.time() - start)


# In[54]:


model.fit(
    X_train,
    y_train,
    epochs=1,
    validation_split=0.05)


# In[55]:


predictions = model.predict(X_test)


# In[56]:


def plot_lstm_prediction(actual, prediction):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Add labels
    plt.ylabel('Price USD')
    plt.xlabel('Trading Days')

    # Plot actual and predicted close values

    plt.plot(actual, '#00FF00', label='Adjusted Close')
    plt.plot(prediction, '#0000FF', label='Predicted Close')

    # Set title
    ax.set_title('Stock Actual vs Prediction')
    ax.legend(loc='upper left')

    plt.show()


# In[57]:


plot_lstm_prediction(y_test,predictions)


# In[58]:


trainScore = model.evaluate(X_train, y_train, verbose=0)
print('Train Score: %.8f MSE (%.8f RMSE)' % (trainScore, math.sqrt(trainScore)))

testScore = model.evaluate(X_test, y_test, verbose=0)
print('Test Score: %.8f MSE (%.8f RMSE)' % (testScore, math.sqrt(testScore)))


# # Improved LSTM Model

# In[59]:


def build_lstm_model_improved(input_dim, output_dim, return_sequences):

    model = Sequential()
    model.add(LSTM(
        input_shape=(None, input_dim),
        units=output_dim,
        return_sequences=return_sequences))

    model.add(Dropout(0.2))

    model.add(LSTM(
        128,
        return_sequences=False))

    model.add(Dropout(0.2))

    model.add(Dense(
        units=1))
    model.add(Activation('linear'))

    return model


# In[61]:


# Set up hyperparameters
batch_size = 100
epochs = 5

# build improved lstm model
model = build_lstm_model_improved(X_train.shape[-1],output_dim =50, return_sequences=True)

start = time.time()
#final_model.compile(loss='mean_squared_error', optimizer='adam')
model.compile(loss='mean_squared_error', optimizer='adam')
print('compilation time : ', time.time() - start)


# In[62]:


model.fit(X_train, 
          y_train, 
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_split=0.05
         )


# In[63]:


predictions = model.predict(X_test, batch_size=batch_size)


# In[64]:


plot_lstm_prediction(y_test,predictions)


# In[65]:


trainScore = model.evaluate(X_train, y_train, verbose=0)
print('Train Score: %.8f MSE (%.8f RMSE)' % (trainScore, math.sqrt(trainScore)))

testScore = model.evaluate(X_test, y_test, verbose=0)
print('Test Score: %.8f MSE (%.8f RMSE)' % (testScore, math.sqrt(testScore)))

