import numpy as np
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def reshape_dataset(data, lags, steps_ahead=1):
    X = []
    Y = []
    steps_ahead -= 1
    for i in range(0, len(data) - lags - steps_ahead):
        r = data[i:i + lags]
        X.append(r)
        Y.append(data[i + lags + steps_ahead])
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


lags = 2
steps_ahead = 1
f = open("denv_colombia.csv")
data = np.loadtxt(f, delimiter=';')
original_data = data[:, 2]
train_data = original_data[:336]
validation_data = original_data[336:418]
test_data = original_data[418:]


X_train, Y_train = reshape_dataset(train_data, lags, steps_ahead)
X_val, Y_val = reshape_dataset(validation_data, lags, steps_ahead)

# reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_val = np.reshape(X_val, (X_val.shape[0], 1, X_val.shape[1]))


model = Sequential()
model.add(LSTM(4, input_dim=lags))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, Y_train, nb_epoch=100, batch_size=1, verbose=2)

trainPredict = model.predict(X_train)
testPredict = model.predict(X_val)

trainScore = math.sqrt(mean_squared_error(Y_train, trainPredict[:, 0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(Y_val, testPredict[:, 0]))
print('Test Score: %.2f RMSE' % (testScore))



# shift train predictions for plotting
trainPredictPlot = np.empty_like([train_data])
trainPredictPlot[:, :] = np.nan
trainPredictPlot[lags:len(trainPredict)+lags, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like([train_data])
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(lags*2)+1:len(train_data)-1, :] = testPredict
# plot baseline and predictions
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
