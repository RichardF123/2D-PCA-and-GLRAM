import numpy as np
import math
import matplotlib.pyplot as plt
#GLRAM core code
def Update_L(A_list, R, k):
    """Update L given R."""
    ML = np.sum([Ai @ R @ R.T @ Ai.T for Ai in A_list], axis=0)
    eigenvalues, eigenvectors = np.linalg.eig(ML)
    idx = np.argsort(eigenvalues)[-k:][::-1]
    return eigenvectors[:, idx]
def Update_R(A_list, L, k):
    """Update R given L."""
    MR = np.sum([Ai.T @ L @ L.T @ Ai for Ai in A_list], axis=0)
    eigenvalues, eigenvectors = np.linalg.eig(MR)
    idx = np.argsort(eigenvalues)[-k:][::-1]
    return eigenvectors[:, idx]
def Als_Algorithm(A_list, init_L, init_R, k1, k2, iterations=100):
    """
    Alternating Least Squares algorithm for optimizing L and R.
    
    Parameters:
    A_list (list of np.ndarray): list of data matrix
    init_L, init_R (np.ndarray): initial L and R
    k1, k2 (int): the number of rows of L and the number of columns of R
    iterations (int): the steps to iteration
    
    Returns:
    L, R (np.ndarray): final L and R
    """
    L = init_L
    R = init_R
    
    for _ in range(iterations):
        L = Update_L(A_list, R, k1)
        R = Update_R(A_list, L, k2)
    return L, R
def Init_Matrix(row,column):
    m,_= np.linalg.qr(np.random.rand(row,column))
    return m

def Normalize(matrice):
    size = matrice[0].shape
    mean = np.zeros(size)
    #caculate the average 
    for ai in matrice:
        mean += ai
    mean /= float(len(matrice))
    for ai in matrice:
        ai = ai - mean
    return matrice , mean

# the RMSRE of GlRAM 
def Frobenius_Norm (Matrix):
    a = np.linalg.norm(Matrix,'fro')
    return a
def Cal_Glram_Rmsre (Matrix_List,L,R):
    sum=0
    num = 0
    for ai in Matrix_List:
        temp = np.dot(np.dot(L,L.T),np.dot(ai,np.dot(R,R.T)))
        num += 1
        error = ai - temp
        sum += Frobenius_Norm (error)**2
    rmsre =   np.sqrt( sum/num)
    return rmsre

import cv2
import os
import numpy as np
Orl_Path = 'D:\download\ORL'
Output_Folder = 'D:\download\ORL_M'
if not os.path.exists(Output_Folder):
    os.makedirs(Output_Folder)
for Person_Folder in os.listdir(Orl_Path):
    Person_Path = os.path.join(Orl_Path, Person_Folder)
    if os.path.isdir(Person_Path):
        for Image_File in os.listdir(Person_Path):
            if Image_File.endswith(".pgm"):  
                Image_Path = os.path.join(Person_Path, Image_File)
                Img = cv2.imread(Image_Path, cv2.IMREAD_GRAYSCALE)                
                if Img is not None:
                    Matrix_File_Path = os.path.join(Output_Folder, f"{Person_Folder}_{Image_File.split('.')[0]}.npy")
                    np.save(Matrix_File_Path, Img)
                    print(f"Image {Image_File} saved as matrix.")
                else:
                    print(f"Failed to read {Image_File}.")

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
Matrices_1,mean = Normalize(Matrices_1)
result = []
for i in range(2,22,2):
    L0 = Init_Matrix(112,i)
    R0 = Init_Matrix(92,i)
    L,R=Als_Algorithm(Matrices_1, L0, R0,i ,i, iterations=120)
    rmsre = Cal_Glram_Rmsre(Matrices_1,L,R)
    result.append(rmsre)
x  = np.arange(2.,22.,2)
plt.plot(x,result)
plt.show
path =  r'D:\download\GLRAM 复现材料\ORL_M\s40_10.npy'
y = np.load(path)
y = y - mean
L0 = Init_Matrix(112,50)
R0 = Init_Matrix(92,50)
L,R=Als_Algorithm(Matrices_1, L0, R0,50,50, iterations=4)
re = np.dot(np.dot(L,L.T),np.dot(y,np.dot(R,R.T)))
plt.imshow(re,cmap = 'gray')


