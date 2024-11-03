import numpy as np
from PIL import Image
def PCA2D(samples, Row_Top):
    '''samples are a list of 2D matrices (images)'''
    # check if the size of all pictures are the same
    assert all(sample.shape == samples[0].shape for sample in samples), "All images must have the same shape."
    
    size = samples[0].shape
    mean = np.zeros(size)
    
    # caculate the average
    for s in samples:
        mean += s
    mean /= float(len(samples))
    
    # caculate the covariance matrix
    Cov_Row = np.zeros((size[1], size[1]))
    for s in samples:
        diff = s - mean
        Cov_Row += np.dot(diff.T, diff)
    Cov_Row /= (float(len(samples)-1))
    
    # caculate the eigenvalues and eigenvectors of the covariance matri
    Row_Eval, Row_Evec = np.linalg.eig(Cov_Row)

    # reorder the eigenvalue
    Sorted_Index_Top = np.argsort(Row_Eval)[::-1][:Row_Top]

    # leave the row_top eigenvectors
    X = Row_Evec[:, Sorted_Index_Top]

    return X
def Restruct(Matrix,X):
    b = []
    for ai in Matrix:
        temp = np.dot(np.dot(ai,m),m.T)
        b.append(temp)
    return b
#caculate the RMSER between two lists of matrice
def Frobenius_Norm (matrix):
    a = np.linalg.norm(matrix,'fro')
    return a
def Calculate_Rmsre(Matrices_1, Matrices_2):
    Total_Squared_Error = 0
    Num_Matrices = 0
    list2 = np.array(Matrices_1) - np.array(Matrices_2)
    for ai in list2:
        Num_Matrices += 1
        Error_Matrix = ai
        Total_Squared_Error += Frobenius_Norm(Error_Matrix)**2  # caculate the norm
    
    # caculate the rmsre
    rmsre = np.sqrt(Total_Squared_Error / Num_Matrices)
    return rmsre
#import orl dataset
import cv2
import os
import numpy as np
# the path of your local dataset
Orl_Path = 'Your local dataset path'
# the path to preserve the matrix of orl
Output_Folder = 'Your local dataset path''
# check whether the folder is existent or not
if not os.path.exists(Output_Folder):
    os.makedirs(Output_Folder)
# ORL has 40 person folder and every person has 10 pictures
for Person_Folder in os.listdir(Orl_Path):
    Person_Path = os.path.join(Orl_Path, Person_Folder)
    if os.path.isdir(Person_Path):
        for Image_File in os.listdir(Person_Path):
            if Image_File.endswith(".pgm"):  # check the picture file is pgm
                Image_Path = os.path.join(Person_Path, Image_File)
                # load the pgm
                Img = cv2.imread(Image_path, cv2.IMREAD_GRAYSCALE)
                
                # check the picture is load correctly
                if Img is not None:
                    #save the picture as a numpy matrix
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
Matrices_1=Add_Matrix_List(Path = 'Your local dataset path')
result = []
for i in range (2,22,2):
    m = PCA2D(Matrices_1, i)
    list = Restruct(Matrices_1,m)
    rmsre = Calculate_Rmsre(Matrices_1, list)
    result.append(rmsre)
x  = np.arange(2.,22.,2)
import matplotlib.pyplot as plt
plt.plot(x,result)
plt.show
# draw a picture of every people
photos = []
for i in range(1, 41):
    Np_Path = f'Your local dataset path'\\s{i}_1.npy'  
    photo = np.load(Np_Path)
    photos.append(photo)  
assert len(photos) == 40, 
fig, axs = plt.subplots(5, 8, figsize=(16, 10))
for row in range(5):
    for col in range(8):
        index = row * 8 + col  
        if index < len(photos):  
            axs[row, col].imshow(photos[index], cmap='gray')  
            axs[row, col].axis('off')  
plt.tight_layout()  
plt.show()
# select the 40th people as the example
path =  r'Your local dataset path'\s40_10.npy'
example = np.load(path)
plt.imshow(example, cmap = 'gray')
m1 = PCA2D(Matrices_1, 30)
path = r'Your local dataset path'\s40_10.npy'
A = np.load(path)
Y = np.dot(A,m1)
def sub_reconstruct(Y, X, k):
    a = Y[:, k]
    b = X[:,k]
    matrix = np.zeros((a.shape[0], X.shape[0]))  # initialize a matrix to save the result
    for i in range(X.shape[0]):
        matrix[:, i] = np.dot(a, b[i]) 
    return matrix
def sum_matrix(Y,X,k):
    a = sub_reconstruct(Y,X,0)
    if k == 0: 
        return a
    else: 
        for i in range(1,k):
            b = sub_reconstruct(Y,X,i)
            a =  np.add(a,b)
    return a
m2 = sum_matrix(Y,m1,5)
m3 = 255 - m2
plt.imshow(m2,cmap = 'gray')
plt.show()
import cv2
v = [x[1] for x in m1]
V = np.mat(v)
res = np.dot(np.dot(Matrices_1[0], V.T), m1.T[[0]])
res = res*255
res = 255 - res
row_im = cv2.imwrite("A0.png",res)
plt.figure(figsize=(10, 10))
plt.imshow(res, cmap='gray') 
plt.title('k=1')
plt.axis('off')  
plt.show()
