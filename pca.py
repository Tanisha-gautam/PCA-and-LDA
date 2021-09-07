import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df_1 = pd.read_csv('pca_data.csv')
df_1= pd.DataFrame(df_1)
df_1_x = df_1.drop(['class'], axis=1)

df_2 = pd.read_csv('lda_data.csv')
df_2= pd.DataFrame(df_2)
df_2_x = df_2.drop(['class'], axis=1)


print("\n \\dataset without class\\\.....\n")
print(df_1_x)
#Part a
#Scatter plot of data samples
print("(a)")

#Scatter plot
fig,ax=plt.subplots(figsize=(14,6) )
plt.grid()
plt.scatter(df_1_x["feature 1"],df_1_x["feature 2"],alpha=0.5, color='r')
plt.title('Scatter plot ', size=18)
plt.xlabel('feature_1', size=12)
plt.ylabel('feature_2', size=12)
plt.show()

fig,ax=plt.subplots(figsize=(12,6))
plt.grid()
plt.scatter(df_2_x["feature 1"],df_2_x["feature 2"],alpha=0.5)
plt.title('Scatter plot ', size=16)
plt.xlabel('feature_1', size=14)
plt.ylabel('feature_2', size=14)
plt.show()


M1= np.mean(df_1_x.T,axis=1)
M2= np.mean(df_2_x.T,axis=1)

C1= df_1_x-M1
C2= df_2_x-M2
print('\n   standardizing the addta....\n')
print(C1)
print(C2)



V1 = np.cov(C1.T)
#Finding the eigen values and eigen vectors
eig_vec1 = np.linalg.eig(V1)[1].T
vals1,vecs1=np.linalg.eig(V1)


explained_variances_1 = []
for i in range(len(vals1)):
    explained_variances_1.append(vals1[i] / np.sum(vals1))
print('\n....variance along both the vectors....\n') 
print(explained_variances_1)


#Plotting eigen directions
fig,ax=plt.subplots(figsize=(12,6))
plt.grid()
plt.scatter(df_1_x["feature 1"],df_1_x["feature 2"],alpha=0.5, color='pink')
plt.arrow(0, 0, *(9*eig_vec1[0]), head_width=0.7, color='red', width=0.2)
plt.arrow(0, 0, *(3*eig_vec1[1]), head_width=0.7, color='b', width=0.2)
plt.title('Scatter plot with eigen directions', size=16)
plt.xlabel('feature_1', size=14)
plt.ylabel('feature_2', size=14)
plt.show()

mat=np.dot(df_1_x,vecs1)

#projecting data in first eigen direction
fig,ax=plt.subplots(figsize=(12,6))
plt.grid()
plt.scatter(df_1_x["feature 1"],df_1_x["feature 2"],alpha=0.5)
plt.quiver([0,0],[0,0],vecs1[0],vecs1[1],angles="xy",scale=3)
plt.scatter(mat[:,0]*vecs1[0][0],mat[:,0]*vecs1[1][0])
plt.title('Projected data in first eigen direction', size=16)
plt.xlabel('feature_1', size=14)
plt.ylabel('feature_2', size=14)
plt.show()

V2 = np.cov(C2.T)

#Finding the eigen values and eigen vectors
eig_vec2 = np.linalg.eig(V2)[1].T
vals2,vecs2=np.linalg.eig(V2)

explained_variances_2 = []
for i in range(len(vals2)):
    explained_variances_2.append(vals2[i] / np.sum(vals2))
print('\n....variance along both the vectors....\n')  
print(explained_variances_2)

#Plotting eigen directions
fig,ax=plt.subplots(figsize=(12,6))
plt.grid()
plt.scatter(df_2_x["feature 1"],df_2_x["feature 2"],alpha=0.5 , color ='green')
plt.arrow(0, 0, *(6*eig_vec2[0]), head_width=0.7, width=0.2, color="r")
plt.arrow(0, 0, *(3*eig_vec2[1]), head_width=0.7, width=0.2)
plt.title('Scatter plot with eigen directions', size=16)
plt.xlabel('feature_1', size=14)
plt.ylabel('feature_2', size=14)
plt.show()

mat_2=np.dot(df_2_x,vecs2)

#projecting data in first eigen direction
fig,ax=plt.subplots(figsize=(12,6))
plt.grid()
plt.scatter(df_2_x["feature 1"],df_2_x["feature 2"],alpha=0.5)
plt.quiver([0,0],[0,0],vecs2[0],vecs2[1],angles="xy",scale=3)
plt.scatter(mat_2[:,0]*vecs2[0][0],mat_2[:,0]*vecs2[1][0])
plt.title('Projected data in first eigen direction', size=16)
plt.xlabel('feature_1', size=14)
plt.ylabel('feature_2', size=14)
plt.show()



