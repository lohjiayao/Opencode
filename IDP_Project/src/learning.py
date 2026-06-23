import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

a = np.array([1, 2, 3, 4, 5])
b = np.array([2, 3, 4, 5, 6])
print("Sum:", a + b)
data = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})
print(data)
plt.plot(a, b)
plt.title("Line Plot")  
plt.xlabel("a")
plt.ylabel("b")
plt.show()

