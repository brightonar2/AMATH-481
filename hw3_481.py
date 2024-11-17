'''
part (a)
'''
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
np.shape(A1)

'''
part (b)
'''
# part B.
# part B.

import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt

L = 4 # domain size
N = 79 # discretization of interior

x = np.linspace(-L, L, N + 2) # add boundary points

dx = x[1] - x[0] # compute dx

P = np.zeros((N, N)) # Compute P matrix

for j in range(N): # build diagonals
    P[j, j] = - 2 - (dx**2) * ( x[j+1] ) ** 2
for j in range(N - 1): # build off-diagonals
    P[j, j + 1] = 1
    P[j + 1, j] = 1

# manually adding boundary conditions
# signs?
P[0,0] += 4/3
P[0,1] += -1/3

P[78,78] += 4/3
P[78,78-1] += -1/3

P = - P / (dx**2)

linL = P # Compute linear operator
D,V = eig(linL) # Compute eigenvalues/eigenvectors

sorted_indices = np.argsort(np.abs(D))[::-1]
Dsort = D[sorted_indices]
Vsort =V[:, sorted_indices]
D5 = Dsort[N-5:N]
V5 = Vsort[:,N-5:N]

# add endpoints to eigenfunctions and normalize result

# using forward and backward differencing relationships:
top = (4/3)*V5[0,:] - (1/3)*V5[1,:]
bottom = (4/3)*V5[78,:] - (1/3)*V5[77,:]

V5_stacked = np.vstack((top,V5,bottom))

V5_normalized = np.zeros_like(V5_stacked)

# normalize!
for i in range(5):
    norm = np.trapz(V5_stacked[:, i]**2, x)
    V5_normalized[:, i] = V5_stacked[:, i] / np.sqrt(norm)

# Eigenvalues corresponding to the normalized eigenfunctions
# A1_rev = D5 / -dx**2  # first five eigenvalues
A1_rev = D5
A3_rev = np.abs(V5_normalized)

A3 = A3_rev[:, ::-1]
A4 = np.flip(A1_rev) # flipped to correct order
A4 = A4.flatten()


# Plot the normalized eigenfunctions (modes) in V5
plt.figure(figsize=(10, 6))
for i in range(5):
    plt.plot(x, A3[:, i], label=f'Mode {i+1}')

plt.legend()
plt.grid(True)
plt.show()

'''
part (c)
'''

#POSITIVE GAMMA
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def shoot2(x_val, y, k, eps):
    return [y[1], (0.05 * (np.abs(y[0]) ** 2) + x_val ** 2 - eps) * y[0]]


tol = 1e-4  # define a tolerance level
col = ['r', 'b', 'g', 'c', 'm', 'k']  # eigenfunction colors
xp = [-2, 2]
xshoot = np.linspace(-2, 2, 41)
k = 1

eps_list = np.array([])  # eigenvalues for this gamma
eig_func = np.empty([41, 2])  # eigenfunctions for this gamma

A_start = 1e-6  # initial guess for amplitude
eps_start = 0.1  # starting value for epsilon

for modes in range(1, 3):  # begin mode loop
    eps = eps_start
    A = A_start
    dA = 0.01  # step size for amplitude

    for _ in range(1000):  # convergence loop for shoot angle
        deps = 0.01

        for _ in range(1000):  # convergence loop for epsilon

            x0 = [A, A * np.sqrt(4 - eps)]

            # Solve differential equation
            sol = solve_ivp(shoot2, xp, x0, t_eval=xshoot, args=(k, eps))
            y = sol.y.T

            end_val = -1 * np.sqrt(4 - eps) * y[-1, 0]  # target boundary condition

            if np.abs(y[-1, 1] - end_val) < tol:  # check convergence
                break

                # Adjust epsilon for convergence
            if np.abs(y[-1, 1] - end_val) > tol:
                if (-1) ** (modes + 1) * (y[-1, 1] - end_val) > 0:
                    eps += deps
                else:
                    eps -= deps / 2
                    deps /= 2

            # Check normalization
        norm = np.trapz(y[:, 0] * y[:, 0], xshoot)
        if np.abs(norm - 1) < tol:
            eps_list = np.append(eps_list, eps)
            break
        else:
            # Adjust amplitude for normalization
            if norm < 1:
                A += dA
            else:
                A -= dA / 2
                dA /= 2

        # Save eigenfunction for current mode
    eps_start = eps + 0.2  # new epsilon start for next mode
    eig_func[:, modes - 1] = np.abs(y[:, 0]) / np.sqrt(norm)
    plt.plot(xshoot, np.abs(y[:, 0]) / np.sqrt(norm), col[modes - 1])  # plot modes

plt.show()

print(eps_list)

A5 = eig_func
A6 = eps_list

# NEGATIVE GAMMA

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def shoot2(x_val, y, k, eps):
    return [y[1], (-0.05 * (np.abs(y[0]) ** 2) + x_val ** 2 - eps) * y[0]]


tol = 1e-4  # define a tolerance level
col = ['r', 'b', 'g', 'c', 'm', 'k']  # eigenfunction colors
xp = [-2, 2]
xshoot = np.linspace(-2, 2, 41)
k = 1

eps_list = np.array([])  # eigenvalues for this gamma
eig_func = np.empty([41, 2])  # eigenfunctions for this gamma

A_start = 1e-6  # initial guess for amplitude
eps_start = 0.1  # starting value for epsilon

for modes in range(1, 3):  # begin mode loop
    eps = eps_start
    A = A_start
    dA = 0.01  # step size for amplitude

    for _ in range(1000):  # convergence loop for shoot angle
        deps = 0.01

        for _ in range(1000):  # convergence loop for epsilon

            x0 = [A, A * np.sqrt(4 - eps)]

            # Solve differential equation
            sol = solve_ivp(shoot2, xp, x0, t_eval=xshoot, args=(k, eps))
            y = sol.y.T

            end_val = -1 * np.sqrt(4 - eps) * y[-1, 0]  # target boundary condition

            if np.abs(y[-1, 1] - end_val) < tol:  # check convergence
                break

                # Adjust epsilon for convergence
            if np.abs(y[-1, 1] - end_val) > tol:
                if (-1) ** (modes + 1) * (y[-1, 1] - end_val) > 0:
                    eps += deps
                else:
                    eps -= deps / 2
                    deps /= 2

            # Check normalization
        norm = np.trapz(y[:, 0] * y[:, 0], xshoot)
        if np.abs(norm - 1) < tol:
            eps_list = np.append(eps_list, eps)
            break
        else:
            # Adjust amplitude for normalization
            if norm < 1:
                A += dA
            else:
                A -= dA / 2
                dA /= 2

        # Save eigenfunction for current mode
    eps_start = eps + 0.2  # new epsilon start for next mode
    eig_func[:, modes - 1] = np.abs(y[:, 0]) / np.sqrt(norm)
    plt.plot(xshoot, np.abs(y[:, 0]) / np.sqrt(norm), col[modes - 1])  # plot modes

plt.show()

print(eps_list)

A7 = eig_func
A8 = eps_list

'''
part (d)
'''
# PART D

import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def shoot2(x_val, x, E):
    return [x[1], ((x_val ** 2) - E) * x[0]]


tol = 1e-4  # define a tolerance level
col = ['r', 'b', 'g', 'c', 'm', 'k']  # eigenfunc colors
E = 1
k = 1
A = np.sqrt(3)
xp = [-2, 2]  # range given in problem
x_span = (-2, 2)  # Generates exactly 41 points from -2 to 2
y0 = [1, A]

# Tolerance values to test
tolerances = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

# To store average step sizes
rk45_step_sizes = []
rk23_step_sizes = []
radau_step_sizes = []
BDF_step_sizes = []

# Run computations for RK45 and RK23
for tol in tolerances:
    options = {'rtol': tol, 'atol': tol}

    # Solve using RK45
    sol_rk45 = solve_ivp(shoot2, x_span, y0, method='RK45', args=(E,), **options)
    avg_step_rk45 = np.mean(np.diff(sol_rk45.t))
    rk45_step_sizes.append(avg_step_rk45)

    # Solve using RK23
    sol_rk23 = solve_ivp(shoot2, x_span, y0, method='RK23', args=(E,), **options)
    avg_step_rk23 = np.mean(np.diff(sol_rk23.t))
    rk23_step_sizes.append(avg_step_rk23)

    # Solve using Radau
    sol_radau = solve_ivp(shoot2, x_span, y0, method='Radau', args=(E,), **options)
    avg_step_radau = np.mean(np.diff(sol_radau.t))
    radau_step_sizes.append(avg_step_radau)

    # Solve using BDF
    sol_BDF = solve_ivp(shoot2, x_span, y0, method='BDF', args=(E,), **options)
    avg_step_BDF = np.mean(np.diff(sol_BDF.t))
    BDF_step_sizes.append(avg_step_BDF)

# Plot on log-log scale
plt.figure(figsize=(10, 5))
plt.loglog(rk45_step_sizes, tolerances, 'o-', label='RK45')
plt.loglog(rk23_step_sizes, tolerances, 'o-', label='RK23')
plt.loglog(radau_step_sizes, tolerances, 'o-', label='radau')
plt.loglog(BDF_step_sizes, tolerances, 'o-', label='BDF')
plt.xlabel("Average Step Size (log scale)")
plt.ylabel("Tolerance (log scale)")
plt.legend()
plt.title("Convergence Study")
plt.show()

# Calculate slopes
slope_rk45 = np.polyfit(np.log10(rk45_step_sizes), np.log10(tolerances), 1)[0]
slope_rk23 = np.polyfit(np.log10(rk23_step_sizes), np.log10(tolerances), 1)[0]
slope_radau = np.polyfit(np.log10(radau_step_sizes), np.log10(tolerances), 1)[0]
slope_BDF = np.polyfit(np.log10(BDF_step_sizes), np.log10(tolerances), 1)[0]

A9 = np.array([slope_rk45, slope_rk23, slope_radau, slope_BDF])

# Save the slopes
print("Slopes for RK45, RK23, Radau, BDF:", A9)

'''
part (e)
'''
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def shoot2(x_val, y, k, eps):
    return [y[1], (k * (x_val ** 2) - eps) * y[0]]


tol = 1e-4  # define a tolerance level
col = ['r', 'b', 'g', 'c', 'm', 'k']  # eigenfunc colors
k = 1  # given
A = 1  # initial guess
xp = [-4, 4]  # range given in problem
# xshoot = np.arange(-4, 4.1, 0.1)
xshoot = np.linspace(-4, 4, 81)

eps_list_new = np.array([])
eig_func_new = np.empty([81, 5])

eps_start = 0  # starting value for epsilon
for modes in range(1, 6):  # begin mode loop
    eps = eps_start  # reset epsilon for each mode
    deps = 0.01  # step size for epsilon
    for _ in range(1000):  # begin convergence loop for epsilon

        x0 = [1, 1 * np.sqrt(16 - eps)]  # guess x[0] = 1 then x[1] is given by initial BC at x = - L

        y = solve_ivp(shoot2, xp, x0, t_eval=xshoot, args=(k, eps))

        end_val = - 1 * np.sqrt(16 - eps) * y.y[0, -1]  # should be target BC

        if abs(y.y[1, -1] - end_val) < tol:  # check for convergence
            eps_list_new = np.append(eps_list_new, eps)
            break  # get out of convergence loop

        if (-1) ** (modes + 1) * (y.y[1, -1] - end_val) > 0:
            eps += deps  # increase epsilon if the last value is positive
        else:
            eps -= deps / 2  # decrease epsilon to refine
            deps /= 2

    eps_start = eps + 0.1  # after finding eigenvalue, pick new start
    norm = np.trapz(y.y[0] * y.y[0], xshoot)  # calculate normalization
    eig_func_new[:, modes - 1] = np.abs(y.y[0]) / np.sqrt(norm)
    plt.plot(xshoot, (y.y[0]) / np.sqrt(norm), col[modes - 1])  # plot modes

plt.show()  # end mode loop
import math
def hermite(xi, n): # using recursion relation to get hermite polynomials
    if n == 0: return 1
    if n == 1: return 2*xi
    return 2*xi*hermite(xi, n-1)-2*(n-1)*hermite(xi,n-2)
    # alt. code in recursion relation for H_2+

def psi(xi, n):
    return np.pi**(-1/4) / np.sqrt(2**n * math.factorial(n)) * hermite(xi,n) * np.exp(-xi**2 / 2)

L = 4 # domain size
xx = np.linspace(-L, L, 81)

# generate exact solutions
t1 = np.abs(psi(xx, 0))
t2 = np.abs(psi(xx, 1))
t3 = np.abs(psi(xx, 2))
t4 = np.abs(psi(xx, 3))
t5 = np.abs(psi(xx, 4))

stacked = np.vstack((t1,t2,t3,t4,t5))
stacked = np.transpose(stacked)

eigen_true = np.array([1,3,5,7,9])

A10 = np.array([])
A11 = np.array([])
A12 = np.array([])
A13 = np.array([])

for i in range(5):
    difference = eig_func_new[:,i] - stacked[:,i]
    difference_norm = np.trapz(difference * difference, xx)
    A10 = np.append(A10,difference_norm)

for i in range(5):
    percent_diff = ( np.abs( eps_list_new[i] - eigen_true[i] ) / eigen_true[i] ) * 100
    A11 = np.append(A11,percent_diff)

for i in range(5):
    difference = A3[:,i] - stacked[:,i]
    difference_norm = np.trapz(difference * difference, xx)
    A12 = np.append(A12,difference_norm)

for i in range(5):
    percent_diff = ( np.abs( A4[i] - eigen_true[i] ) / eigen_true[i] ) * 100
    A13 = np.append(A13,percent_diff)


print(A6,A8)
