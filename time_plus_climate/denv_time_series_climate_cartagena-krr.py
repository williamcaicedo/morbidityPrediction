import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler


def reshape_dataset(data, lags, steps_ahead=1):
    X = []
    Y = []
    #lags += 1
    steps_ahead -= 1
    for i in range(0, len(data) - lags - steps_ahead):
        r = data[i:i + lags]
        X.append(r)
        Y.append(data[i + lags + steps_ahead])
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


def augment_dataset(data, climate_data, lags):
    X = np.column_stack((data, climate_data[:len(data)]))
    for i in range(1, lags):
        X = np.column_stack((X, np.roll(climate_data[:len(data)], -2*i)))
    return X


lags = 2
steps_ahead = 4
f = open("denv_cartagena+climate_data.csv")
data = np.loadtxt(f, delimiter=',')
original_data = data[:, 2]
train_data = original_data[:418]
#validation_data = original_data[336:418]
test_data = original_data[418:]
xscaler = StandardScaler()
yscaler = StandardScaler()
#train_data = scaler.fit_transform(train_data)
X_train, Y_train = reshape_dataset(train_data, lags, steps_ahead)
X_test, Y_test = reshape_dataset(test_data, lags, steps_ahead)

climate_data = data[:, [4, 5]]
X_train = augment_dataset(X_train, climate_data, lags)
X_train = xscaler.fit_transform(X_train)
Y_train = yscaler.fit_transform(Y_train)

np.set_printoptions(threshold=np.nan)
#print(np.column_stack((X_train, Y_train)))


week_data = [' ' for i in range(len(train_data))]
for i in range(len(train_data)):
    if i % 8 == 0:
        week_data[i] = str(int(data[i, 0])) + '-' + str(int(data[i, 1]))

X_train_weeks, Y_train_weeks = reshape_dataset(week_data, lags, steps_ahead)

kr = KernelRidge(kernel='rbf', gamma=1e-5, alpha=1e-7)

training_size = int(X_train.shape[0]*0.5)
validation_size = X_train.shape[0] - training_size
j = training_size
validation_predictions = np.zeros(validation_size)
for i in range(validation_size):
    kr.fit(X_train[i:j], Y_train[i:j])
    validation_predictions[i] = kr.predict(np.array([X_train[j]]))
    j += 1
validation_predictions = yscaler.inverse_transform(validation_predictions)


Y_train = yscaler.inverse_transform(Y_train)
#mape = np.mean(np.abs((Y_train[training_size:] - validation_predictions) / Y_train[training_size:])) * 100
print("Explained Variance: {0}".format(metrics.explained_variance_score(Y_train[training_size:], validation_predictions)))
r2 = metrics.r2_score(Y_train[training_size:], validation_predictions)
print("R^2: {0}".format(r2))
mae = metrics.mean_absolute_error(Y_train[training_size:], validation_predictions)
print("Mean Absolute Error: {0}".format(mae))
fig = plt.figure()
plt.plot(validation_predictions, label='predicted')
plt.plot(Y_train[training_size:], label='observed')
plt.xticks(range(validation_size), Y_train_weeks[training_size:], rotation='vertical', size='small')

plt.title('Gaussian KRR dengue model ({0} lag(s), {2} step(s) ahead, MAE: {1:.3f})'.format(lags, mae, steps_ahead))
plt.legend(loc="upper right")
plt.tight_layout()
fig.savefig('Gaussian Kernel Ridge Regression Cartagena (4 steps ahead).png', format='png', dpi=500)
plt.show()


#Random walk
j = training_size
validation_predictions = np.zeros(validation_size)
for i in range(validation_size):
    validation_predictions[i] = Y_train[j-steps_ahead]
    j += 1
print("Random walk results:")
print("Explained Variance: {0}".format(metrics.explained_variance_score(Y_train[training_size:], validation_predictions)))
r2 = metrics.r2_score(Y_train[training_size:], validation_predictions)
print("R^2: {0}".format(r2))
mae = metrics.mean_absolute_error(Y_train[training_size:], validation_predictions)
print("Mean Absolute Error: {0}".format(mae))
plt.plot(validation_predictions, label='predicted')
plt.plot(Y_train[training_size:], label='observed')
plt.xticks(range(validation_size), Y_train_weeks[training_size:], rotation='vertical', size='small')

plt.title('Random Walk dengue model (MAE: {0:.3f})'.format(mae))
plt.legend(loc="upper right")
plt.show()
