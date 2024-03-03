# In this example we're demonstrating the gradient descent algorithm on 1D function
# the function we're working on is f(x), which has a gradient f`(x)

# For visualization visit: https://www.desmos.com/calculator
#     - write f(x), f`(x), ...etc into the equations section

import random

def function(x):
    return (x-5) ** 2

def function_derivative(x):
    return 2 * (x-5)

def gradient_descent(function, function_derivative, start, max_iterations=100, step=0.05):
    print(f'Gradient Descent Algorithm with {{`step`: {step}, `max_iteration`: {max_iterations}}}')
    new_point = start
    for iteration in range(max_iterations):
        point = new_point
        delta = -step * function_derivative(point)
        new_point = point + delta
        print(f'iteration: {iteration: 4d} >> derivative: {function_derivative(point): 10.2f}, delta: {delta: 10.4f}, point: {point: 10.2f} -> {new_point: 10.2f}, function={function(new_point): 10.2f}')
        # if the new_point has a small change that is less than a threshold epsilon, we can skip iterating to the `max_iterations`
        if abs(delta) < 0.0001:
            break

if __name__ == '__main__':
    start_point = random.randint(-1000, 1000) # choose any value either specific or random
    gradient_descent(function, function_derivative, start_point)
    
