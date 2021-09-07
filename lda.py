import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder


df_2 = pd.read_csv('lda_data.csv')
df_2= pd.DataFrame(df_2)
df_2_x = df_2.drop(['class'], axis=1)

Class=np.array(["1","2"])
class_feature_means = pd.DataFrame(columns=Class)
for c, rows in df_2.groupby('class'):
    class_feature_means[c] = np.mean(rows, axis=0)
Mean=class_feature_means.dropna(axis=1)[:2].copy()
print(Mean)

within_class_scatter_matrix = np.zeros((2,2))
for c, rows in df_2.groupby('class'):
    rows = rows.drop(['class'], axis=1)
    
    s = np.zeros((2,2))
for index, row in rows.iterrows():
        x, mc = row.values.reshape(2,1), Mean[c].values.reshape(2,1)
        
        s += (x - mc).dot((x - mc).T)
    
        within_class_scatter_matrix += s
        
feature_means = df_2_x.mean()
between_class_scatter_matrix = np.zeros((2,2))
for c in Mean:    
    n = len(df_2.loc[df_2['class'] == c].index)
    
    mc, m = Mean[c].values.reshape(2,1), feature_means.values.reshape(2,1)
    
    between_class_scatter_matrix += n * (mc - m).dot((mc - m).T)

eigen_values, eigen_vectors = np.linalg.eig(np.linalg.inv(within_class_scatter_matrix).dot(between_class_scatter_matrix))

pairs = [(np.abs(eigen_values[i]), eigen_vectors[:,i]) for i in range(len(eigen_values))]
pairs = sorted(pairs, key=lambda x: x[0], reverse=True)
for pair in pairs:
    print(pair[0])

eigen_value_sums = sum(eigen_values)
print('Explained Variance')
for i, pair in enumerate(pairs):
    print('Eigenvector {}: {}'.format(i, (pair[0]/eigen_value_sums).real))
    
w_matrix = np.hstack((pairs[0][1].reshape(2,1), pairs[1][1].reshape(2,1))).real

X_lda = np.array(df_2_x.dot(w_matrix))


le = LabelEncoder()
y = le.fit_transform(df_2['class'])

plt.xlabel('LD1')
plt.ylabel('LD2')
plt.scatter( X_lda[:,0],df_2['class'],c=y,cmap='rainbow',alpha=0.7,)

fig,ax=plt.subplots(figsize=(12,6))
plt.grid()

plt.scatter(X_lda[:,0],X_lda[:,1],c=y)
plt.title('Projected data in first eigen direction', size=16)
plt.xlabel('feature_1', size=14)
plt.ylabel('feature_2', size=14)
plt.show()

