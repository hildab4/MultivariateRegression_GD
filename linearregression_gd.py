'''
    Linear Regression Model using Gradient Descent

    Hilda Beltrán Acosta
    A01251916
'''

# Import necessary libraries to implement linear regression with GD
import numpy
import pandas as pd
# Matplotlib and sklearn metrics are only used to visualize and test
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Define parameters, learning rate and an empty array for errors
params = [0, 0, 0, 0]
params_initial = params
lr = 0.01
errorsf = []

# Load data into DataFrame
columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
df1 = pd.read_csv('iris.data', names = columns)

# Create extra column and set it to 1, in order to have a value to multiply the bias
df1['bias'] = 1

# Random order for the DataFrame rows, since most of the biggest values of y are at the end
df = df1.sample(frac = 1)

# Divide between inputs and outputs, as well as training and testing
x = df[['bias', 'sepal length', 'sepal width', 'petal length']][:125].to_numpy()
y = df['petal width'][:125].to_numpy()
x_test = df[['bias', 'sepal length', 'sepal width', 'petal length']][126:].to_numpy()
y_real = df['petal width'][126:].to_numpy()

# y_pred is used at the end to validate the linear regression model
y_pred = numpy.zeros((x_test.shape[0]))

'''
    Function h_get calculates the hypothetical value, relying in the values
    that are set in params.

    @param params, x
    @return vertical sum of the prediction matrix
'''
def h_get(params, x):
    return ((params * x).sum(axis = 1))

'''
    Function errors calculate the difference between the predicted and the
    real value for y.

    @param h, y
    @return difference of h and y
'''
def errors(h, y):
    error = (h - y)
    return error

'''
    Function update is where the final errors are calculated, which are
    helpful for updating the parameters.

    @param x, y, lr, params
    @return array of updated params
'''
def update(x, y, lr, params):
    h = 0; error = 0
    h = h_get(params, x)
    error = errors(h, y)
    error = (error * x.T).sum(axis = 1)
    params = params - (lr * 1/len(x)) * error
    
    return params

'''
    Function gd calls the function update, in order to update parameters.
    The times it calls update is defined by the user with the # of epochs.
    Function show_errors is used to visualize the error.

    @param epochs, x, params
    @return updated parameters
'''
def gd(epochs, x, params):
    for i in range (epochs):
        params = update(x, y, lr, params)
        show_errors(params, x, y)
    return params

'''
    Function show_errors is used to append errors into an array, in order to
    visualize them later with matplotlib

    @param params, x, y
    @return
'''
def show_errors(params, x, y):
    h = 0; error = 0; global errorsf
    h = h_get(params, x)
    error = errors(h, y) ** 2
    error = error.sum() / 2 * len(x)
    errorsf.append(error)

# Save the updated parameters obtained by calling the gd function with 2000 epochs
params = gd(2000, x, params)

# Plotting the errors from the show_errors function
plt.plot(errorsf)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.show()

# Predictions
# Use the updated parameters to predict some values for the test arrays
for i in range (x_test.shape[0]):
    y_pred[i] = params[0] + params[1] * x_test[i][1] + params[2] * x_test[i][2] + params[3] * x_test[i][3]

# r2_score is a function from sklearn used to validate the accuracy of our model
# it doesn't affect in any other way during the implementation of the model.
r2 = r2_score(y_real, y_pred)

# Print data for user to understand what's happening
print('Initial parameters')
print(params_initial)
print('\nUpdated parameters')
print(params)
print('\nR2 coefficient between prediction and real value')
print(r2)