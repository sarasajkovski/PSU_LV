import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

def non_func(x):
    return 1.6345 - 0.6235*np.cos(0.6067*x) - 1.3501*np.sin(0.6067*x) - \
           1.1622*np.cos(2*x*0.6067) - 0.9443*np.sin(2*x*0.6067)

def add_noise(y):
    np.random.seed(14)
    varNoise = np.max(y) - np.min(y)
    return y + 0.1 * varNoise * np.random.normal(0, 1, len(y))

n_samples = 50  
x = np.linspace(1, 10, n_samples)
y_true = non_func(x)
y_measured = add_noise(y_true)

x = x[:, np.newaxis]
y_measured = y_measured[:, np.newaxis]

np.random.seed(12)
indices = np.random.permutation(len(x))
train_size = int(0.7 * len(x))
indeksi_train = indices[:train_size]
indeksi_test = indices[train_size:]

xtrain_orig = x[indeksi_train]
ytrain = y_measured[indeksi_train]

xtest_orig = x[indeksi_test]
ytest = y_measured[indeksi_test]

degrees = [2, 6, 15]
MSEtrain = []
MSEtest = []

plt.figure(figsize=(10, 6))

x_plot = np.linspace(1, 10, 500)[:, np.newaxis]
y_plot_true = non_func(x_plot)

plt.plot(x_plot, y_plot_true, 'k--', label='Pozadinska funkcija f(x)')

# Prolazimo kroz sve stupnjeve
for deg in degrees:
    poly = PolynomialFeatures(degree=deg)
    
    xtrain = poly.fit_transform(xtrain_orig)
    xtest = poly.transform(xtest_orig)
    x_plot_poly = poly.transform(x_plot)

    model = LinearRegression()
    model.fit(xtrain, ytrain)

    ytrain_pred = model.predict(xtrain)
    ytest_pred = model.predict(xtest)

    mse_train = mean_squared_error(ytrain, ytrain_pred)
    mse_test = mean_squared_error(ytest, ytest_pred)

    MSEtrain.append(mse_train)
    MSEtest.append(mse_test)

    y_plot_pred = model.predict(x_plot_poly)
    plt.plot(x_plot, y_plot_pred, label=f'degree {deg}')

plt.plot(xtrain_orig, ytrain, 'o', label='Train podaci', color='gray', alpha=0.6)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Modeli različitog stupnja vs stvarna funkcija')
plt.legend()
plt.grid(True)
plt.show()

print('MSE na skupu za učenje:', MSEtrain)
print('MSE na test skupu:', MSEtest)
