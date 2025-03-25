import numpy as np

a = np.array([[1,2,3],[4,5,6]])
print(a)
print(a[1,1])

x = np.array([[[1,2,3],[3,4,5]],[[5,6,7],[9,10,11]]])
print(x[1,1,[1]])
Z = np.zeros(5)
print(Z)
np.shape(Z)
Z2 = np.zeros((4,5))
print(Z2)
np.shape(Z2)
Y = np.ones((2,3))
print(Y)
F = np.full((7,8),11)

X = np.linspace(0,5,10)
print(X)
X2 = np.arange(0,5,0.2)
print(X2)

a = 1
b = 6
amount = 50
nopat = np.random.randint(a,b+1,amount)
print(nopat)
x = np.random.randn(100)
print(x)

a = np.array([[1,5],[6,4]])
b = np.array([[5,6],[7,8]])

add = a + b
print(add)

substraction = a - b
print(substraction)

divide = a/b
print(divide)

mul = a*b
print(mul)

matmul = a@b
print(matmul)

Y = a >= 1
print(Y)

Y = a <= 1
print(Y)


print(np.sin(a))#for sin value of a

print(np.cos(a))#for cos value of a

print(np.tan(a))#for tan value of a

print(np.sqrt(a))#for sqrt value of a

addition = a + 2
print(addition)

Substraction = a - 1
print(Substraction)

n = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
n1 = np.reshape(n,(3,4))
rows,cols = n1.shape
print(rows,cols)