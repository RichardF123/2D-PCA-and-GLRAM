import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from collections import Counter
# flaten the matrix to a vector
def Flaten_Matrix(Matrix):
    m,n = Matrix.shape
    Vector_New = np.zeros(m*n)
    num = 0 
    for j in range(n):
        for i in range(m):
            Vector_New[num] = Matrix[i][j]
            num += 1
    return Vector_New
#stack vectors as a new matrix
def Trans_Stack(Matrices):
    M = Flaten_Matrix(Matrices[0])
    for ai in Matrices[1:]:
        m = Flaten_Matrix(ai)
        M = np.vstack((M,m))
    return M.T    
def Add_Matrix_List(Path):
    Matrix_List = []
    for j in range(1,41):
        for i in range(1,11):
            Np_Path = f"{Path}\s{j}_{i}.npy"
            Matrix = np.load(Np_Path)
            Matrix_List.append(Matrix)
            print(f"add matrix{i} successfully")
    print('Add complete')
    return Matrix_List
Matrices_1=Add_Matrix_List(Path = 'D:\download\GLRAM 复现材料\ORL_M')
def Normalize(matrice):
    size = matrice[0].shape
    mean = np.zeros(size)
    for ai in matrice:
        mean += ai
    mean /= float(len(matrice))
    for ai in matrice:
        ai = ai - mean
    return matrice , mean
# get eigenface
N_Matrices_1,Mean = Normalize(Matrices_1)
plt.imshow(Mean,cmap = 'gray')
test_M = Trans_Stack(N_Matrices_1)
path =  r'D:\download\GLRAM 复现材料\ORL_M\s40_10.npy'
y = np.load(path)
plt.imshow(y, cmap = 'gray')
y1 = y-Mean
plt.imshow(y1,cmap = 'gray')
U,s,Vh= np.linalg.svd(test_M,full_matrices=False)
k = 20  # the number of eigenface you choose
eigenfaces = U[:, :k]
eigenfaces.shape
v = Flaten_Matrix(y1)@eigenfaces
u = np.dot(eigenfaces,v.T)
plt.imshow(u.reshape([92,112]).T+Mean,cmap = 'gray')
