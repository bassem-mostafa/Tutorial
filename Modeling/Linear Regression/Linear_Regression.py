# In this example we're demonstrating the linear regression using gradient descent algorithm on 1D function

# refer to `https://github.com/bassem-mostafa/Tutorial/blob/release/Optimization/Gradient%20Descent/Gradient_Descent.py` for more information gradient descent optimization algorithm

from matplotlib import pyplot as plt
import numpy as np

def function(a, b, x):
    return a + b * x

def linear_regression():
    # number of epochs
    n_epochs = 1000 # An epoch is complete whenever every point has been already used for computing the loss

    # Model parameters random initialization
    a = np.random.randn(1)
    b = np.random.randn(1)
    
    for epoch in range(n_epochs):
        # Computes our model's predicted output
        yhat = function(a, b, x_train)
        
        # How wrong is our model? That's the error! 
        error = (y_train - yhat)
        
        # It is a regression, so it computes mean squared error (MSE)
        loss = (error ** 2).mean()
        
        # Computes gradients for both "a" and "b" parameters
        # refer to `https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e`
        #     - Gradient Descent: step 2 Compute the Gradients
        a_grad = -2 * error.mean()
        b_grad = -2 * (x_train * error).mean()
        
        # Gradient Descent learning rate
        lr= 0.1
        
        # Updates parameters using gradients and the learning rate
        a = a - lr * a_grad
        b = b - lr * b_grad
        print(f'epoch {epoch}: a: {a}, b: {b}')
    return a, b

if __name__ == '__main__':
    print('Linear Regression Demo')
    # random seed initialization
    # np.random.seed(42)

    # Data generation
    x = np.random.rand(100, 1) # input
    a, b = 1, 2   # real parameters value of `a`, `b` that would be estimated using linear regression
    y = function(a, b, x) # real value of `y`
    mu, sigma = 0, 0.1      # gaussian noise mean, sigma
    noise = sigma * np.random.randn(*y.shape) + mu # gaussian noise
    y = y + noise # `y` added noise
    
    # Shuffles the indices
    idx = np.arange(100)
    np.random.shuffle(idx)
    
    # Uses first 80 random indices for train
    train_idx = idx[:80]
    # Uses the remaining indices for validation
    val_idx = idx[80:]
    
    # Generates train and validation sets
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    
    # visualize data generated
    fig = plt.figure('Linear Regression Demo')
    ax = fig.subplots(1, 2)
    ax[0].set_title('train data')
    ax[0].scatter(x[train_idx], y[train_idx])
    ax[1].set_title('validation data')
    ax[1].scatter(x[val_idx], y[val_idx])

    fig.show()
    plt.show(block=False)
    plt.pause(0.5)
    
    a_hat, b_hat = linear_regression()
    print(f'\nActual a: {a}, b: {b}')
    print(f'Estimated a: {a_hat}, b: {b_hat}')
    
    plt.waitforbuttonpress()