# In this example we're demonstrating the linear interpolation algorithm on 2D function
# the function we're working on is f(x, y)

import numpy as np

def linear_kernel(length):
    if length < 2: raise RuntimeError('length MUST be greater than or equal 2')
    kernel = np.zeros((length, 2))
    step = 1 / (length - 1)
    kernel[0, :] = [1, 0]
    for i in range(1, length):
        kernel[i, :] = [kernel[i-1, 0] - step, kernel[i-1, 1] + step]
    return kernel

# Reference https://en.wikipedia.org/wiki/Bilinear_interpolation#Repeated linear interpolation
def interpolate_bilinear(input, kernel):
    interpolate_x = np.matmul(kernel[0], input)
    return np.matmul(interpolate_x, kernel[1])

if __name__ == '__main__':
    print('Bi-Linear Interpolation Demo')
    input = np.array([[1, 5], [5, 9]])                          # 2x2
    kernel = linear_kernel(5), np.transpose(linear_kernel(5))   # 5x2, 2x5
    output = interpolate_bilinear(input, kernel)                # 5x5
    print(f'\ninterpolation input: \n{input}')
    print(f'\ninterpolation kernel x: \n{kernel[0]}')
    print(f'\ninterpolation kernel y: \n{kernel[1]}')
    print(f'\ninterpolation output: \n{output}')
