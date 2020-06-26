import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

print(cv.__version__)
matrix = np.arange(21).reshape(3, 7)
print(matrix)
fig, axis = plt.subplots()
axis.plot([1,2,3,4], [4,7,8,4])
plt.show()
