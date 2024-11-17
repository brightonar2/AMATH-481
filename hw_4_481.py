import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags

# part A. laplacian

m = 8    # N value in x and y directions
n = m * m  # total size of matrix

e0 = np.zeros((n, 1))  # vector of zeros
e1 = np.ones((n, 1))   # vector of ones
e2 = np.copy(e1)    # copy the one vector
e4 = np.copy(e0)    # copy the zero vector

for j in range(1, m+1):
    e2[m*j-1] = 0  # overwrite every m^th value with zero
    e4[m*j-1] = 1  # overwirte every m^th value with one

# Shift to correct positions
e3 = np.zeros_like(e2)
e3[1:n] = e2[0:n-1]
e3[0] = e2[n-1]

e5 = np.zeros_like(e4)
e5[1:n] = e4[0:n-1]
e5[0] = e4[n-1]

# Place diagonal elements
diagonals = [e1.flatten(), e1.flatten(), e5.flatten(),
             e2.flatten(), -4 * e1.flatten(), e3.flatten(),
             e4.flatten(), e1.flatten(), e1.flatten()]
offsets = [-(n-m), -m, -m+1, -1, 0, 1, m-1, m, (n-m)]

matA = spdiags(diagonals, offsets, n, n).toarray()

L = 10

dx = (L - (-L)) / 8

A1 = matA / dx**2

# Plot matrix structure
plt.figure(5)
plt.spy(matA)
plt.title('Matrix Structure')
plt.show()




# Part b first partial in x
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags

# part b. First derivative in x

m = 8    # N value in x and y directions
n = m * m  # total size of matrix

# Place diagonal elements
diagonals = [e1.flatten(), -e1.flatten(), e0.flatten(), e1.flatten(), -e1.flatten()]
offsets = [-(n-m),-m, 0, m, (n-m)]

matA = spdiags(diagonals, offsets, n, n).toarray()

A2 = matA / (2*dx)


# Plot matrix structure
plt.figure(5)
plt.spy(A2)
plt.title('Matrix Structure for First x Partial Derivative')
plt.show()




# part x first partial in y
import matplotlib.pyplot as plt
from scipy.sparse import spdiags

# part c. First derivative in y
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags

# part A. laplacian

m = 8    # N value in x and y directions
n = m * m  # total size of matrix

e0 = np.zeros((n, 1))  # vector of zeros
e1 = np.ones((n, 1))   # vector of ones
e2 = np.copy(e1)    # copy the one vector
e4 = np.copy(e0)    # copy the zero vector

for j in range(1, m+1):
    e2[m*j-1] = 0  # overwrite every m^th value with zero
    e4[m*j-1] = 1  # overwirte every m^th value with one

# Shift to correct positions
e3 = np.zeros_like(e2)
e3[1:n] = e2[0:n-1]
e3[0] = e2[n-1]

e5 = np.zeros_like(e4)
e5[1:n] = e4[0:n-1]
e5[0] = e4[n-1]

# Place diagonal elements
diagonals = [e5.flatten(),
             -e2.flatten(), e0.flatten(), e3.flatten(),
             -e4.flatten()]
offsets = [ -m+1, -1, 0, 1, m-1]

matA = spdiags(diagonals, offsets, n, n).toarray()

A3 = matA / (2*dx)

# Plot matrix structure
plt.figure(5)
plt.spy(matA)
plt.title('Matrix Structure')
plt.show()



print(A1)
print(A2)
print(A3)