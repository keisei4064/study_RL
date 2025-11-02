import numpy as np
if not hasattr(np, "int"):
    np.int = int # pyright: ignore[reportAttributeAccessIssue]
from dezero import Variable
import dezero.functions as F

# Inner products
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
a, b = Variable(a), Variable(b)  # Optional
c = F.matmul(a, b)
print("vector inner product:")
print(c)
print()

# Matrix product
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
c = F.matmul(a, b)
print("matrix product:")
print(c)