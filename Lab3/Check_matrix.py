import numpy as np

A = np.loadtxt("A_matrix.txt", skiprows=1)
B = np.loadtxt("B_matrix.txt", skiprows=1)
C = np.loadtxt("Result.txt", skiprows=1)

R = A @ B

if (R == C).all():
    print("Matrix multiplication is correct.")
else:
    print("Matrix multiplication is wrong.")