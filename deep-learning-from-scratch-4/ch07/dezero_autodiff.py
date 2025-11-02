import numpy as np

if not hasattr(np, "int"):
    np.int = int  # pyright: ignore[reportAttributeAccessIssue]
from dezero import Variable

x_np = np.array(5.0)
x = Variable(x_np)

y = 3 * x**2
print(y)

print("autodiff")
y.backward()
print(x.grad)
