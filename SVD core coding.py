import numpy as np
import matplotlib.pyplot as plt

def SVD(A):
    from scipy.linalg import svd
    U,s,Vh= svd(A,full_matrices=False)
    return U,s,Vh
#restruct data
def Restruct(U,s,Vh,k):
    s_truncated = np.diag(s[:k])
    Approx = np.dot(U[:,:k],np.dot(s_truncated,Vh[:k,:]))
    return Approx
#Do SVD on a list of matrices 
def Matrices_Svd(Matrices_Old,d):
    Matrices_New = []
    for ai in Matrices_Old:
        u,s,vh = SVD(ai)
        A = Restruct(u,s,vh,d)
        Matrices_New.append(A)
    return Matrices_New
#caculate the RMSER between two lists of matrice
def Frobenius_Norm (matrix):
    a = np.linalg.norm(matrix,'fro')
    return a
def Calculate_Svd_Rmsre(matrices1, matrices2):
    Total_Squared_Error = 0
    Num_Matrices = 0
    list2 = np.array(matrices1) - np.array(matrices2)
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
Output_Folder = 'Your local dataset path'
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
Result = []
for i in range(2,22,2):
    list = Matrices_Svd(Matrices_1,i)
    rmsre = Calculate_Svd_Rmsre(Matrices_1, list)
    result.append(rmsre)
x  = np.arange(2.,22.,2)
plt.plot(x,result)
plt.show




 "nbformat": 4,
 "nbformat_minor": 5
}
