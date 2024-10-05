import numpy as np

# Problem 1:
# Method 1. Newton
x0 = -1.6
x = [x0]
threshold = 10 ** (-6)


def func(x):
    return x * np.sin(3 * x) - np.exp(x)


def dfunc(x):
    return np.sin(3 * x) + (3 * x * np.cos(3 * x)) - np.exp(x)


n = 0
for i in range(100):

    fx0 = func(x0)  # Evaluate the function at x0
    dfx0 = dfunc(x0)  # Evaluate the derivative at x0
    n += 1
    x1 = x0 - fx0 / dfx0
    x.append(x1)
    x0 = x1
    if abs(fx0) <= threshold:
        break

# Method 2. Bisection

int_0 = np.array([-0.7, -0.4])
x_bi = np.array([int_0[0], np.mean(int_0), int_0[1]])

mid = [x_bi[1]]

N = 1  # confusion about source of error -- is it iterates or matrix size?

'''
for i in range(100):
    N += 1
    if func(x_bi[0]) * func(x_bi[1]) < 0:
            x_bi = np.array([x_bi[0], np.mean([x_bi[0], x_bi[1]]), x_bi[1]])
    elif func(x_bi[1]) * func(x_bi[2]) < 0:
           x_bi = np.array([x_bi[1], np.mean([x_bi[1], x_bi[2]]), x_bi[2]])
    mid.append(x_bi[1])
    if abs(func(x_bi[1])) <= threshold:
        break
'''

for i in range(100):

    if abs(func(x_bi[1])) > threshold:
        if func(x_bi[0]) * func(x_bi[1]) < 0:
            x_bi = np.array([x_bi[0], np.mean([x_bi[0], x_bi[1]]), x_bi[1]])
        elif func(x_bi[1]) * func(x_bi[2]) < 0:
            x_bi = np.array([x_bi[1], np.mean([x_bi[1], x_bi[2]]), x_bi[2]])
        N += 1
        mid.append(x_bi[1])

    elif abs(func(x_bi[1])) <= threshold:
        break

A1 = x
A2 = mid
A3 = np.array([n, N])
# A3 = A3.reshape(1, 2)

# Problem 2:
A = np.array([[1, 2], [-1, 1]])
B = np.array([[2, 0], [0, 2]])
C = np.array([[2, 0, -3], [0, 0, -1]])
D = np.array([[1, 2], [2, 3], [-1, 0]])
x_new = np.array([[1], [0]])
y = np.array([[0], [1]])
z = np.array([[1], [2], [-1]])

A4 = A + B
A5 = np.ravel(3 * x_new - 4 * y)
A6 = np.ravel(np.dot(A, x_new))
A7 = np.ravel(np.dot(B, x_new - y))
A8 = np.ravel(np.dot(D, x_new))
A9 = np.ravel(np.dot(D, y) + z)
A10 = np.dot(A, B)
A11 = np.dot(B, C)
A12 = np.dot(C, D)