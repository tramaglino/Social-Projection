import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.cluster import KMeans
import seaborn as sns

####### PLOTTING HIGH DIMENSIONAL DATASET ########

grezzo = pd.read_csv('DatixAnalisi.csv')
df_meno1colonne = grezzo.drop(columns=['Unnamed: 0', 'Name'])

# centrare e scalare i dati
sc_df = preprocessing.scale(df_meno1colonne)
X = pd.DataFrame(sc_df)

#### DOING KMEANS ON THE FIRST THREE PRINCIPAL COMPONENTS OF PCA ####
#Kmeans
pca = PCA(n_components=2).fit(sc_df)
data2D = pca.transform(X)
plt.figure()
plt.scatter(data2D[:,0], data2D[:,1])
plt.show()
plt.close()

km = KMeans(n_clusters=5)
km.fit(X)
centers2D = pca.transform(km.cluster_centers_)
plt.hold(True)
labels=np.array([km.labels_])
print(labels)


k_rng = range(1,100)
sse = [] # errore residuo
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df)
    sse.append(km.inertia_)
# ora ho le varianze in sse, devo plottarle per trovare il gomito

plt.xlabel('N of clusters')
plt.ylabel('Sum squared error')

plt.plot(k_rng,sse, color = 'purple')
#plt.savefig('Number of clusters.png', bbox_inches='tight')
plt.show()
plt.close('all')

km = KMeans(n_clusters=13)
y_predicted = km.fit_predict(df)
df['Cluster'] = y_predicted

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(df.loc[:,0], df.loc[:,1], df.loc[:,2], c =km.labels_.astype(float))
fig.show()

sns.scatterplot(data = df, x = 0, y = 1, hue='Cluster', style = 'Cluster', palette = 'colorblind')

##3d scatter
sns.set(style = "darkgrid")
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
x = df[0]
y = df[1]
z = df[2]

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")

ax.scatter(x, y, z, c=km.labels_.astype(float))

plt.show()

###STA FACENDO ESATTAMENTE Quello che voglio fare io###
sentence_list=["Hi how are you", "Good morning" ...] #i have 10 setences

km = KMeans(n_clusters=5, init='k-means++',n_init=10, verbose=1)
#with 5 cluster, i want 5 different colors
km.fit(vectorized)
km.labels_ # [0,1,2,3,3,4,4,5,2,5]

pca = PCA(n_components=2).fit(X)
data2D = pca.transform(X)
plt.scatter(data2D[:,0], data2D[:,1])

km.fit(X)
centers2D = pca.transform(km.cluster_centers_)
plt.hold(True)
labels=np.array([km.labels_])
print(labels)