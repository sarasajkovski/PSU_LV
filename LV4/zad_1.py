import numpy as np
import sklearn.linear_model as lm

linearModel = lm.LinearRegression()

linearModel.fit(xtrain,ytrain)

ytest_pred = linearModel.predict(xtest)