import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model



d = np.array([[18, 7, 14, 31, 21, 5, 11, 16, 26, 29] , 
              [55, 17, 36, 85, 62, 18, 33, 41, 63, 87]])

d = d.transpose()

print(d)


plt.scatter(d[:,0],d[:,1])
plt.show()


#On peut effectivement soupçonner une relation linéaire entre les xi et les yi


X = np.array([[18,  7, 14, 31, 21,  5, 11, 16, 26, 29]]).reshape(-1,1)
Y = d[:,1].reshape(-1,1)




reg = linear_model.LinearRegression()
reg.fit(X,Y)


a = reg.coef_[0][0]

b = reg.intercept_[0]



print(a)


#le coefficient de regression vaut 2,73


print(b)


#L'intercept vaut 1,02


x = [18,  7, 14, 31, 21,  5, 11, 16, 26, 29]
y = [a*i + b for i in x]


y


plt.plot(x,y, c='red')
plt.scatter(d[:,0],d[:,1], c = 'green')
plt.show()

#Une estimation plausible de Y à xi = 21 est :
print(a*21+b)


#L'écart observé entre la valeur estimée et la valeur réelle est :
print(62 - a*21+b)

#Cet écart est appelé le résidu en 21

#???

