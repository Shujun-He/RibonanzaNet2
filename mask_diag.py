import numpy as np
import matplotlib.pyplot as plt

def set_near_diagonal_to_nan(matrix, k=4):
    assert matrix.shape[0] == matrix.shape[1], "Matrix must be square"
    n = matrix.shape[0]
    mask = np.abs(np.arange(n)[:, None] - np.arange(n)) < k
    matrix[mask] = np.nan
    return matrix


A = np.random.rand(10, 10)
masked = set_near_diagonal_to_nan(A.copy(), k=4)
print("Original:\n", A)
print("Masked:\n", masked)

plt.subplot(1, 2, 1)
plt.imshow(A)
plt.subplot(1, 2, 2)
plt.imshow(masked)
plt.savefig("masked_diag.png")