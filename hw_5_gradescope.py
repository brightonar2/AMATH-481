import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.sparse import spdiags

# Precursor -- chop up + build matrices

# defining system parameters and temporal domain
tp = [0,4]
tspan = np.arange(0, 4.5, 0.5)
nu = 0.001
Lx, Ly = 20, 20
nx, ny = 64, 64
N = nx * ny # 4096

# Define spatial domain and initial conditions
x2 = np.linspace(-Lx/2, Lx/2, nx + 1)
x = x2[:nx]
y2 = np.linspace(-Ly/2, Ly/2, ny + 1)
y = y2[:ny]
X, Y = np.meshgrid(x, y)

w = 1 * np.exp(- X**2 - Y**2 / 20) + 1j * np.zeros((nx, ny))  # Initialize as complex, defined on mesh grid!
# shape: (64,64) --> contains

nu = 0.001

dx = x[1] - x[0]
dy = y[1] - y[0]


# build matrices!

m = nx    # N value in x and y directions

e0 = np.zeros((N, 1))  # vector of zeros
e1 = np.ones((N, 1))   # vector of ones
e2 = np.copy(e1)    # copy the one vector
e4 = np.copy(e0)    # copy the zero vector

for j in range(1, m+1):
    e2[m*j-1] = 0  # overwrite every m^th value with zero
    e4[m*j-1] = 1  # overwirte every m^th value with one

# Shift to correct positions
e3 = np.zeros_like(e2)
e3[1:N] = e2[0:N-1]
e3[0] = e2[N-1]

e5 = np.zeros_like(e4)
e5[1:N] = e4[0:N-1]
e5[0] = e4[N-1]

# Place diagonal elements
diagonals = [e1.flatten(), e1.flatten(), e5.flatten(),
             e2.flatten(), -4 * e1.flatten(), e3.flatten(),
             e4.flatten(), e1.flatten(), e1.flatten()]
offsets = [-(N-m), -m, -m+1, -1, 0, 1, m-1, m, (N-m)]

matA = spdiags(diagonals, offsets, N, N).toarray()

A = matA / dx**2 # use in part (a)

A_n = matA.copy()
A_n[0,0] = 2
A_n = A_n / dx**2 # use in part (b) to avoid singular

# part b. First derivative in x

# Place diagonal elements
diagonals = [e1.flatten(), -e1.flatten(), e0.flatten(), e1.flatten(), -e1.flatten()]
offsets = [-(N-m),-m, 0, m, (N-m)]

matA = spdiags(diagonals, offsets, N, N).toarray()

B = matA / (2*dx)

# part c. deriv in y

for j in range(1, m+1):
    e2[m*j-1] = 0  # overwrite every m^th value with zero
    e4[m*j-1] = 1  # overwirte every m^th value with one

# Shift to correct positions
e3 = np.zeros_like(e2)
e3[1:N] = e2[0:N-1]
e3[0] = e2[N-1]

e5 = np.zeros_like(e4)
e5[1:N] = e4[0:N-1]
e5[0] = e4[N-1]

# Place diagonal elements
diagonals = [e5.flatten(),
             -e2.flatten(), e0.flatten(), e3.flatten(),
             -e4.flatten()]
offsets = [ -m+1, -1, 0, 1, m-1]

matA = spdiags(diagonals, offsets, N, N).toarray()

C = matA / (2*dx)

# part A

# Define spectral k values
kx = (2 * np.pi / Lx) * np.concatenate((np.arange(0, nx / 2), np.arange(-nx / 2, 0)))
kx[0] = 1e-6
ky = (2 * np.pi / Ly) * np.concatenate((np.arange(0, ny / 2), np.arange(-ny / 2, 0)))
ky[0] = 1e-6
KX, KY = np.meshgrid(kx, ky)
K = KX ** 2 + KY ** 2

wt0 = np.hstack([np.real(w.reshape(N)), np.imag(w.reshape(N))])  # separate fourier transform of initial


# condition into real and imaginary components
# size: (8192,)

# Define the ODE system
def spc_rhs(t, wt2, nx, ny, N, KX, KY, K, nu, A, B, C):
    w2c = wt2[:N] + 1j * wt2[N:]
    w2c_g = w2c.reshape((nx, ny))

    wt = fft2(w2c_g)

    # REVIEW SIGN HERE
    psit = -wt / K  # solve psi in fourier domain

    # IFT 1
    psi = (ifft2(psit)).reshape(N)  # return psi to position space; reshape to N column vector (4096,)
    # IFT 2
    w = (ifft2(wt)).reshape(N)  # return stream function to position space; reshape to N column vector (4096,)

    # Time evolve using derivative matrices. This returns updated w!
    rhs = nu * A.dot(w) - B.dot(psi) * C.dot(w) + C.dot(psi) * B.dot(w)
    rhs = rhs.reshape((nx, ny))

    return np.hstack([np.real(rhs.reshape(N)), np.imag(rhs.reshape(N))])  # returns new vorticy as (8192,)


# for plotting
w_all = []
rows = int(np.ceil(len(tspan) / 3))
cols = 3

# solve the ODE and plot the results!
wtsol = solve_ivp(spc_rhs, tp, wt0, method='RK45', t_eval=tspan, args=(nx, ny, N, KX, KY, K, nu, A, B, C))
wtsol = wtsol.y.T

A1 = (wtsol[:, :N]).T

for j, t in enumerate(tspan):
    w = np.real(wtsol[j, :N].reshape((nx, ny)))
    plt.subplot(rows, cols, j + 1)
    plt.pcolor(x, y, w, shading='auto')
    plt.title(f'Time: {t}')
    plt.colorbar()

plt.tight_layout()
plt.show()

print(A1[0, 0])

# Part (b) A\b

from scipy.linalg import solve

w = 1 * np.exp(- X ** 2 - Y ** 2 / 20) + 1j * np.zeros((nx, ny))  # Initialize as complex, defined on mesh grid!
# shape: (64,64) --> contains

wt0 = np.hstack([np.real(w.reshape(N)), np.imag(w.reshape(N))])
A_n = A
A_n[0, 0] = 2


# Define the ODE system
def spc_rhs(t, wt2, nu, A, B, C):
    wc = wt2[0:N] + 1j * wt2[N:]  # rewrite fourier transformed IC input as real + im

    # direct solve for psi
    psi = solve(A, wc)

    # Time evolve using derivative matrices. This returns updated w!
    rhs = nu * A.dot(wc) - B.dot(psi) * C.dot(wc) + C.dot(psi) * B.dot(wc)

    return np.hstack([np.real(rhs), np.imag(rhs)])  # returns new vorticy as (8192,)


# for plotting
w_all = []
rows = int(np.ceil(len(tspan) / 3))
cols = 3

# solve the ODE and plot the results!
wtsol = solve_ivp(spc_rhs, tp, wt0, method='RK45', t_eval=tspan, args=(nu, A_n, B, C))
wtsol = wtsol.y.T

A2 = (wtsol[:, :N]).T

for j, t in enumerate(tspan):
    w = np.real(wtsol[j, :N].reshape((nx, ny)))
    plt.subplot(rows, cols, j + 1)
    plt.pcolor(x, y, w, shading='auto')
    plt.title(f'Time: {t}')
    plt.colorbar()

plt.tight_layout()
plt.show()
print(A2[0,0])

# part C: LU factorization

from scipy.linalg import solve
from scipy.linalg import lu, solve_triangular

w = 1 * np.exp(- X ** 2 - Y ** 2 / 20) + 1j * np.zeros((nx, ny))  # Initialize as complex, defined on mesh grid!
# shape: (64,64) --> contains

wt0 = np.hstack([np.real(w.reshape(N)), np.imag(w.reshape(N))])


# Define the ODE system
def spc_rhs(t, wt2, nu, A, B, C):
    wc = wt2[0:N] + 1j * wt2[N:]

    P, L, U = lu(A)
    Pb = np.dot(P, wc)
    y = solve_triangular(L, Pb, lower=True)
    psi = solve_triangular(U, y)

    # Time evolve using derivative matrices. This returns updated w!
    rhs = nu * A.dot(wc) - B.dot(psi) * C.dot(wc) + C.dot(psi) * B.dot(wc)

    return np.hstack([np.real(rhs), np.imag(rhs)])  # returns new vorticy as (8192,)


# for plotting
w_all = []
rows = int(np.ceil(len(tspan) / 3))
cols = 3

# solve the ODE and plot the results!
wtsol = solve_ivp(spc_rhs, tp, wt0, method='RK45', t_eval=tspan, args=(nu, A_n, B, C))
wtsol = wtsol.y.T

A3 = (wtsol[:, :N]).T

for j, t in enumerate(tspan):
    w = np.real(wtsol[j, :N].reshape((nx, ny)))
    plt.subplot(rows, cols, j + 1)
    plt.pcolor(x, y, w, shading='auto')
    plt.title(f'Time: {t}')
    plt.colorbar()

plt.tight_layout()
plt.show()
print(A3[0,0])