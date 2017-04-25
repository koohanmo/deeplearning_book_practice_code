import numpy as np

# Scalars
s = 10

# Vectors
V = np.array([0, 1, 2, 3, 4, 5, 6, 7])
S= [1,3,5]
#print (V[S])


# Matrices
M1 = np.matrix([[1,2],[3,4],[5,6]])
#print(M1)
#print (M1[0,:])
#print (M1[:,1])


# Tensor
T = np.array([[[1,2],[3,4],[5,6]],
              [[1,2],[3,4],[5,6]],
              [[1,2],[3,4],[5,6]]])
#print (T.shape)


# Transpose

# Matrices_Transpose
M1 = np.matrix([[1,2],[3,4],[5,6]])
#print(M1.transpose())


# Tensor_Transpose
T = np.array([[[1,2],[3,4],[5,6]],
              [[1,2],[3,4],[5,6]],
              [[1,2],[3,4],[5,6]]])
#print (T.transpose())


 # Broadcasting
A = np.matrix([[1,2],[3,4],[5,6]])
b = np.array([10,10])
print(A+b)


# Matmul
A = np.matrix([[1,2],[3,4]])
B = np.matrix([[4,5],[6,7]])

print(np.matmul(A,B))
print(np.dot(A,B))


# Norm
V= np.array([1,2,3,4])
print (np.linalg.norm(V,ord=1)) #L1 norm
print (np.linalg.norm(V,ord=2)) #L2 norm

M = np.matrix([[1,2],[3,4],[5,6]])
print (np.linalg.norm(M,'fro'))


# Diagonal
V = np.array([1,2,3,4])
D = np.diag(V)
print(D)
print(np.linalg.inv(D))

