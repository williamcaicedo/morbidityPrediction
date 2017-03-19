import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import sklearn.metrics as metrics

import matplotlib.pyplot as plt
def show_prediction_curve(clf, cv, X, y):

    fig = plt.figure()
    all_predictions = []
    for train,test in cv.split(X):
        clf.fit(X[train], y[train])
        predictions = clf.predict(X[test])
        all_predictions = np.append(all_predictions, predictions)

    all_predictions = y_scaler.inverse_transform(all_predictions)
    y = y_scaler.inverse_transform(y)
    r2 = metrics.r2_score(y, all_predictions)
    print("R^2: {0}".format(r2))

    print("Explained Variance: {0}".format(
        metrics.explained_variance_score(y, all_predictions)))
    mae = metrics.mean_absolute_error(y, all_predictions)
    print("Mean Absolute Error: {0}".format(mae))
    #fig = plt.figure()
    plt.plot(all_predictions, label='predicted')
    plt.plot(y, label='observed')
    #plt.xticks(range(validation_size), Y_train_weeks[training_size:], rotation='vertical', size='small')

    plt.title('Gaussian KRR dengue model (MAE: {0:.3f})'.format(mae))
    plt.legend(loc="upper right")
    plt.tight_layout()
    #fig.savefig('Gaussian Kernel Ridge Regression Cartagena (climatic).png', format='png', dpi=500)
    plt.show()

weeks_before = -5

f = open("datos_clima.csv")
data = np.loadtxt(f, delimiter=',')
train_data = data[:321, :]
test_data = data[321:, :]

x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train = train_data[:weeks_before, [3, 6]]
y_train = np.roll(train_data[:, 0], weeks_before)[:weeks_before]

X_train = x_scaler.fit_transform(X_train)
y_train = y_scaler.fit_transform(y_train)

print(len(X_train))
print(len(y_train))
model = KernelRidge(kernel='rbf', gamma=0.1)
cv = KFold(n_splits=5)
grid_search = GridSearchCV(model, cv=5, n_jobs=-1,
                           param_grid={"alpha": np.linspace(1e-15, 1, 100),
                                       "gamma": np.linspace(1e-15, 5, 100)})
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
model.set_params(**{'alpha': grid_search.best_params_['alpha'], 'gamma': grid_search.best_params_['gamma']})
show_prediction_curve(model, cv, X_train, y_train)

