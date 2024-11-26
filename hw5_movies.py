# movie making ᕙ(▀̿̿ĺ̯̿̿▀̿ ̿) ᕗ

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2
from scipy.integrate import odeint

import imageio.v2 as imageio  # Updated import for v2 behavior
import os

# set up!

output_dir = "frames"
os.makedirs(output_dir, exist_ok=True)

# Define parameters
tspan = np.arange(0, 25.5, 0.5)
nu = 0.001
Lx, Ly = 20, 20
nx, ny = 128, 128
N = nx * ny

# Define spatial domain and initial conditions
x2 = np.linspace(-Lx/2, Lx/2, nx + 1)
x = x2[:nx]
y2 = np.linspace(-Ly/2, Ly/2, ny + 1)
y = y2[:ny]
X, Y = np.meshgrid(x, y)

# Define spectral k values
kx = (2 * np.pi / Lx) * np.concatenate((np.arange(0, nx/2), np.arange(-nx/2, 0)))
kx[0] = 1e-6
ky = (2 * np.pi / Ly) * np.concatenate((np.arange(0, ny/2), np.arange(-ny/2, 0)))
ky[0] = 1e-6
KX, KY = np.meshgrid(kx, ky)
K = KX**2 + KY**2

# Define the ODE system
def spc_rhs(wt2, t, nx, ny, N, KX, KY, K, nu):
    wtc = wt2[0:N] + 1j*wt2[N:]
    wt = wtc.reshape((nx, ny))
    psit = -wt / K
    psix = np.real(ifft2(1j * KX * psit))
    psiy = np.real(ifft2(1j * KY * psit))
    wx = np.real(ifft2(1j * KX * wt))
    wy = np.real(ifft2(1j * KY * wt))
    rhs = (-nu * K * wt + fft2(wx * psiy - wy * psix)).reshape(N)
    return np.hstack([np.real(rhs),np.imag(rhs)])


# A. Two oppositely charged gaussians next to each other
w = (1 * np.exp(-(X + 1) ** 2 - (Y) ** 2) + 1j * np.zeros((nx, ny))) - (
            1 * np.exp(-(X - 1) ** 2 - (Y + 1) ** 2) + 1j * np.zeros((nx, ny)))

# Solve the ODE and plot the results
wt0 = np.hstack([np.real(fft2(w).reshape(N)), np.imag(fft2(w).reshape(N))])
wtsol = odeint(spc_rhs, wt0, tspan, args=(nx, ny, N, KX, KY, K, nu))

# reshape wtsol
wtsol_s = np.zeros((len(tspan), nx, ny), dtype=np.complex128)

# need to inverse fourier transform each row of output wtsol to get out of fourier space for animation!
for i, wt in enumerate(wtsol):
    w_real = wt[:N].reshape((nx, ny))
    w_imag = wt[N:].reshape((nx, ny))

    w_complex = w_real + 1j * w_imag

    w_spatial = np.real(ifft2(w_complex))

    wtsol_s[i] = w_spatial

# Generate frames
for j, t in enumerate(tspan):
    w = np.real(wtsol_s[j, :N].reshape((nx, ny)))
    plt.figure(figsize=(6, 5))
    plt.pcolor(x, y, w, shading='auto', cmap='magma')
    plt.colorbar(label="Vorticity")
    plt.title(f"Time: {t:.2f}")
    plt.xlabel("x")
    plt.ylabel("y")
    frame_path = os.path.join(output_dir, f"frame_{j:03d}.png")
    plt.savefig(frame_path)
    plt.close()

# Create a GIF
gif_path = "vorticity_evolution_A.gif"
with imageio.get_writer(gif_path, mode='I', duration=0.5) as writer:
    for j in range(len(tspan)):
        frame_path = os.path.join(output_dir, f"frame_{j:03d}.png")
        image = imageio.imread(frame_path)
        writer.append_data(image)

# ༼ ༎ຶ ෴ ༎ຶ༽
print(f"Animation saved as {gif_path}")

# B. Two similarly charged gaussians next to each other
w = (1 * np.exp(-(X + 1) ** 2 - (Y) ** 2) + 1j * np.zeros((nx, ny))) + (
            1 * np.exp(-(X - 1) ** 2 - (Y + 1) ** 2) + 1j * np.zeros((nx, ny)))

# Solve the ODE and plot the results
wt0 = np.hstack([np.real(fft2(w).reshape(N)), np.imag(fft2(w).reshape(N))])
wtsol = odeint(spc_rhs, wt0, tspan, args=(nx, ny, N, KX, KY, K, nu))

# reshape wtsol
wtsol_s = np.zeros((len(tspan), nx, ny), dtype=np.complex128)

# need to inverse fourier transform each row of output wtsol to get out of fourier space for animation!
for i, wt in enumerate(wtsol):
    w_real = wt[:N].reshape((nx, ny))
    w_imag = wt[N:].reshape((nx, ny))

    w_complex = w_real + 1j * w_imag

    w_spatial = np.real(ifft2(w_complex))

    wtsol_s[i] = w_spatial

# Generate frames
for j, t in enumerate(tspan):
    w = np.real(wtsol_s[j, :N].reshape((nx, ny)))
    plt.figure(figsize=(6, 5))
    plt.pcolor(x, y, w, shading='auto', cmap='magma')
    plt.colorbar(label="Vorticity")
    plt.title(f"Time: {t:.2f}")
    plt.xlabel("x")
    plt.ylabel("y")
    frame_path = os.path.join(output_dir, f"frame_{j:03d}.png")
    plt.savefig(frame_path)
    plt.close()

# Create a GIF
gif_path = "vorticity_evolution_B.gif"
with imageio.get_writer(gif_path, mode='I', duration=0.5) as writer:
    for j in range(len(tspan)):
        frame_path = os.path.join(output_dir, f"frame_{j:03d}.png")
        image = imageio.imread(frame_path)
        writer.append_data(image)

# ༼ ༎ຶ ෴ ༎ຶ༽
print(f"Animation saved as {gif_path}")

# C. Two colliding pairs
w = (1 * np.exp(-(X + 2) ** 2 - (Y) ** 2) + 1j * np.zeros((nx, ny)) - 1 * np.exp(
    -(X + 2) ** 2 - (Y + 1) ** 2) + 1j * np.zeros((nx, ny))) + -1 * ((
            1 * np.exp(-(X - 2) ** 2 - (Y) ** 2) + 1j * np.zeros((nx, ny)) - 1 * np.exp(
        -(X - 2) ** 2 - (Y + 1) ** 2) + 1j * np.zeros((nx, ny))))

# Solve the ODE and plot the results
wt0 = np.hstack([np.real(fft2(w).reshape(N)), np.imag(fft2(w).reshape(N))])
wtsol = odeint(spc_rhs, wt0, tspan, args=(nx, ny, N, KX, KY, K, nu))

# reshape wtsol
wtsol_s = np.zeros((len(tspan), nx, ny), dtype=np.complex128)

# need to inverse fourier transform each row of output wtsol to get out of fourier space for animation!
for i, wt in enumerate(wtsol):
    w_real = wt[:N].reshape((nx, ny))
    w_imag = wt[N:].reshape((nx, ny))

    w_complex = w_real + 1j * w_imag

    w_spatial = np.real(ifft2(w_complex))

    wtsol_s[i] = w_spatial

# Generate frames
for j, t in enumerate(tspan):
    w = np.real(wtsol_s[j, :N].reshape((nx, ny)))
    plt.figure(figsize=(6, 5))
    plt.pcolor(x, y, w, shading='auto', cmap='magma')
    plt.colorbar(label="Vorticity")
    plt.title(f"Time: {t:.2f}")
    plt.xlabel("x")
    plt.ylabel("y")
    frame_path = os.path.join(output_dir, f"frame_{j:03d}.png")
    plt.savefig(frame_path)
    plt.close()

# Create a GIF
gif_path = "vorticity_evolution_C.gif"
with imageio.get_writer(gif_path, mode='I', duration=0.5) as writer:
    for j in range(len(tspan)):
        frame_path = os.path.join(output_dir, f"frame_{j:03d}.png")
        image = imageio.imread(frame_path)
        writer.append_data(image)

# ༼ ༎ຶ ෴ ༎ຶ༽
print(f"Animation saved as {gif_path}")

# D. Random assortment!

# generate a random collection of vortices:

# number
num_vortices = 15  # choose between 10 - 15

# set field start
w = np.zeros((nx, ny), dtype=np.complex128)

for i in range(num_vortices):
    # randomize
    x0 = np.random.uniform(-10, 10)  # X position
    y0 = np.random.uniform(-10, 10)  # Y position
    strength = np.random.uniform(0.5, 2.0)  # Amplitude
    charge = np.random.choice([-1, 1])  # Charge (positive or negative)
    ellipticity = np.random.uniform(0.5, 1.5)  # Ellipticity factor

    # make gaussian
    gaussian_1 = charge * strength * np.exp(-ellipticity * ((X - x0) ** 2 + (Y - y0) ** 2))

    # add it!
    w += gaussian_1 + 1j * np.zeros_like(gaussian_1)

# Solve the ODE and plot the results
wt0 = np.hstack([np.real(fft2(w).reshape(N)), np.imag(fft2(w).reshape(N))])
wtsol = odeint(spc_rhs, wt0, tspan, args=(nx, ny, N, KX, KY, K, nu))

# reshape wtsol
wtsol_s = np.zeros((len(tspan), nx, ny), dtype=np.complex128)

# need to inverse fourier transform each row of output wtsol to get out of fourier space for animation!
for i, wt in enumerate(wtsol):
    w_real = wt[:N].reshape((nx, ny))
    w_imag = wt[N:].reshape((nx, ny))

    w_complex = w_real + 1j * w_imag

    w_spatial = np.real(ifft2(w_complex))

    wtsol_s[i] = w_spatial

# Generate frames
for j, t in enumerate(tspan):
    w = np.real(wtsol_s[j, :N].reshape((nx, ny)))
    plt.figure(figsize=(6, 5))
    plt.pcolor(x, y, w, shading='auto', cmap='magma')
    plt.colorbar(label="Vorticity")
    plt.title(f"Time: {t:.2f}")
    plt.xlabel("x")
    plt.ylabel("y")
    frame_path = os.path.join(output_dir, f"frame_{j:03d}.png")
    plt.savefig(frame_path)
    plt.close()

# Create a GIF
gif_path = "vorticity_evolution_D.gif"
with imageio.get_writer(gif_path, mode='I', duration=0.5) as writer:
    for j in range(len(tspan)):
        frame_path = os.path.join(output_dir, f"frame_{j:03d}.png")
        image = imageio.imread(frame_path)
        writer.append_data(image)

# ༼ ༎ຶ ෴ ༎ຶ༽
print(f"Animation saved as {gif_path}")