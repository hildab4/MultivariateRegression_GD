# Multivariate Regression Model using Gradient Descent

Implemented a Multivariate Regression Model using Gradient Descent, using three inputs ('sepal length', 'sepal width', 'petal length') in order to predict an output value for petal width. This implementation runs 2000 epochs, even though this can be changed in a function's parameter. After predicting some values with the test data, an R2 coefficient is calculated to validate the Regression Model is working correctly.

# SMA0401A
## Implements a Machine Learning's algorithm or technique, without using any framework, this could be regressions, trees, clusters, etc... 
The Multivariate Regression Model using Gradient Descent is implemented without using any framework, the only libraries used for this implementation are NumPy and Pandas to extract and manipulate the data. There are other libraries such as Matplotlib and Sklearn Metrics to visualize the errors and measure the accuracy of the model, they weren't used during the implementation of Gradient Descent algorithm nor the Multivariate Regression Model.

# Files
## Dataset
The Dataset used for this implementation is 'iris.data', which can be found in this GitHub Repository. In order to change the Dataset there are some modifications that need to be done in the script before executing it; such as changing the loading file for the DataFrame and changing the columns name from the DataFrame that are going to be used.
