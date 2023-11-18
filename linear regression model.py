import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("C:/Users/dulsh/Desktop/example/Book1.csv")
print(df)

plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df.area, df.price)
plt.show()

reg = linear_model.LinearRegression()
reg.fit(df[['area']],df.price)

print(reg.predict([[5000]]))
