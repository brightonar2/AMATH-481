import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def shoot2(x, x_val, k, eps):
    return [x[1], (k * (x_val ** 2) - eps) * x[0]]


tol = 1e-4  # define a tolerance level
col = ['r', 'b', 'g', 'c', 'm', 'k']  # eigenfunc colors
k = 1  # given
A = 1  # initial guess
xp = [-4, 4]  # range given in problem
xshoot = np.arange(-4, 4.1, 0.1)

eps_list = np.array([])
eig_func = np.empty([81, 5])

eps_start = 0  # starting value for epsilon
for modes in range(1, 6):  # begin mode loop
    eps = eps_start  # reset epsilon for each mode
    deps = 0.01  # step size for epsilon
    for _ in range(1000):  # begin convergence loop for epsilon
        x0 = [1, 1 * np.sqrt(16 - eps)]  # guess x[0] = 1 then x[1] is given by initial BC at x = - L

        y = odeint(shoot2, x0, xshoot, args=(k, eps))

        end_val = - 1 * np.sqrt(16 - eps) * y[-1, 0]  # should be target BC

        if abs(y[-1, 1] - end_val) < tol:  # check for convergence
            eps_list = np.append(eps_list, eps)
            break  # get out of convergence loop

        if (-1) ** (modes + 1) * (y[-1, 1] - end_val) > 0:
            eps += deps  # increase epsilon if the last value is positive
        else:
            eps -= deps / 2  # decrease epsilon to refine
            deps /= 2

    eps_start = eps + 0.1  # after finding eigenvalue, pick new start
    norm = np.trapz(y[:, 0] * y[:, 0], xshoot)  # calculate normalization
    eig_func[:, modes - 1] = np.abs( y[:, 0] / np.sqrt(norm) )
    plt.plot(xshoot, y[:, 0] / np.sqrt(norm), col[modes - 1])  # plot modes

plt.show()  # end mode loop

A1 = eig_func
A2 = eps_list

print(eig_func)