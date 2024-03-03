# In this example we're demonstrating the linear interpolation algorithm on 1D function
# the function we're working on is f(x)

import numpy as np

def linear_kernel(length):
    if length < 2: raise RuntimeError('length MUST be greater than or equal 2')
    kernel = np.zeros((length, 2))
    step = 1 / (length - 1)
    kernel[0, :] = [1, 0]
    for i in range(1, length):
        kernel[i, :] = [kernel[i-1, 0] - step, kernel[i-1, 1] + step]
    return kernel

def interpolate_linear(input, kernel):
    return np.matmul(kernel, input)

if __name__ == '__main__':
    print('Linear Interpolation Demo')
    input = np.array([1, 5])                    # 2x1
    kernel = linear_kernel(5)                   # Nx2
    output = interpolate_linear(input, kernel)  # Nx1
    print(f'\ninterpolation input: \n{input}')
    print(f'\ninterpolation kernel: \n{kernel}')
    print(f'\ninterpolation output: \n{output}')
