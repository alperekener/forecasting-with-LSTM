
# importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler

# data extraction
dataset= pd.read_excel('cruise-statistics.xlsx','by-months',usecols=['#cruise_ships'])
dataset = np.array(dataset).reshape(-1,1)
scaler = MinMaxScaler()
dataset= scaler.fit_transform(dataset)


# create our train and test sets with the get_data function
def get_data(dataset):
   
    train_size = 96

    train = dataset[0:train_size]
    test = dataset[train_size:len(dataset)]
    
    # Train Data
    train_x = []
    train_y = []

    for i in range(0, train_size - look_back - output_size):
        x = dataset[i:(i+look_back)]
        y = dataset[(i+look_back):(i+look_back+output_size)]
        train_x.append(np.reshape(x, (1, look_back)))
        train_y.append(y)

    train_x = np.array(train_x)
    train_y = np.array(train_y)


    # Test Data
    test_x = []
    test_y = []
   
    for i in range(train_size, (len(dataset) - output_size - look_back)):
        x = dataset[i:(i+look_back)]
        y = dataset[(i + look_back):(i + look_back + output_size)]
        test_x.append(np.reshape(x, (1, look_back)))
        test_y.append(y)

    test_x = np.array(test_x)
    test_y = np.array(test_y)


    # Forecast Data
    f_data = []
    x = dataset[-look_back:]
    f_data.append(np.reshape(x, (1, look_back)))
    f_data = np.array(f_data)

    return train_x, train_y, test_x, test_y, f_data


output_size = 2 
epochs = 700
look_back = 3

x_train, y_train, x_test, y_test, f_data = get_data(dataset)

y_test = y_test.reshape(y_test.shape[0],2)
y_train = y_train.reshape(y_train.shape[0],2)


# Model
model = Sequential()
model.add(LSTM(50, input_shape=(1, look_back), return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(7))
model.add(Dense(output_size))
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=epochs, verbose=0)

score = model.evaluate(x_test, y_test)
print("%2s: %.2f%%" % (model.metrics_names[1], score[1]*100))
print("%2s: %.2f%%" % (model.metrics_names[0], score[0]*100))

print(model.summary())

# predection
predicted = model.predict(f_data)
predicted = scaler.inverse_transform(predicted)
print("Predicted Next Values: ", predicted)

# to compare
y_pred = model.predict(x_test)
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)

# for graph
plt.figure(figsize=(14,5))
plt.plot(y_test[:,0], label= 'real number of passengers')
plt.plot(y_pred[:,0], label= 'predicted number of passengers')
plt.legend()
print(plt.show())





