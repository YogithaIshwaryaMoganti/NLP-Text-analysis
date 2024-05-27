import os
import nltk
import time
from datetime import datetime
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans


#Read the data set
df = pd.read_csv('dailykos.csv')
print(df.head(3))
print(df.shape)


#Compute Euclidean distances
distances = pdist(df, metric='euclidean')
dist = linkage(distances, method='ward') 

#Create a dendrogram 
plt.figure(figsize=(15, 10))
dendrogram(dist, leaf_font_size=8, leaf_rotation=45)
plt.title('Dendrogram')
plt.xlabel('Observations')
plt.ylabel('Distance')
plt.show()

'''

Determining the appropriate number of clusters depends on a variety of factors, taking into account the specific attributes 
of the data at hand. In the case of handling new articles or blog posts, they can be grouped into separate categories based 
on their content, which might include subjects such as sports, entertainment, and politics. Once these categories have been 
defined, one can employ different techniques like the elbow method and silhouette score to identify the optimal cluster quantity.

'''

# Performed hierarchical clustering with 7 clusters
hc_clusters = fcluster(dist, 7, criterion='maxclust')
print(np.unique(hc_clusters))

# Create a new DataFrame with the cluster assignments 
df_clusters = pd.DataFrame({'Cluster': hc_clusters}) 
df_clusters = pd.concat([df_clusters, df], axis=1)
df_clusters

counts = df_clusters['Cluster'].value_counts()
print(counts)


'''
Q) How many observations are in cluster 3?
A) There are 803 observations in the cluster 3(output from below output)
'''

print('Number of observation in cluster 3 are: ', sum(hc_clusters == 3))

'''
Q) Which cluster has the most observations?
A) Cluster 2 has most observations from below.
'''

max_cluster = counts.idxmax()
print('Most observations are in cluster: ', max_cluster)

'''
Q) Which cluster has fewest observations?
A) Cluster 5 has less observations from below.
'''

min_cluster = counts.idxmin()
print('Fewest observations are in cluster: ', min_cluster)

'''
Q) Instead of looking at the average value in each variable individually, we’ll just look at the top 6 words in each cluster. 
Compute the mean frequency values of each of the words in cluster 1, and 
then output the 6 words that occur the most frequently.
A) In cluster number 1, the word 'November' stands out as the most commonly occurring word when considering the average values.

'''

# Filter the data for Cluster 1
c1 = df_clusters[hc_clusters == 1]
mean_freq = c1.mean()
# Sort the words by their mean frequencies in descending order for top 6 words
sorted_words = mean_freq.sort_values(ascending=False)[:6]
sorted_words.values
print(sorted_words)

'''
Q)Now repeat the command given in the previous problem for each of the other clusters, and answer the following questions.
'''

## Top 6 words in all 7 clusters

top_cluster_words_df = pd.DataFrame()
var_names = [f"cluster_{i}" for i in range(1, 8)]

for i in range(1,8):
    cluster = df_clusters[hc_clusters == i]
    mean_freq = cluster.mean()
    top_words = mean_freq.sort_values(ascending=False)[:6] 
    print('Top words in Cluster: ',i,' are: \n',top_words ) 
    cluster_name = var_names[i-1]
    freq_name = f"{var_names[i-1]}_frequency"
    top_cluster_words_df[cluster_name] = top_words.index.tolist()
    top_cluster_words_df[freq_name] = top_words.values.tolist()

print(top_cluster_words_df)

'''
Q) Which cluster could best be described as the cluster related to the Iraq war?
A) Based on the prominent words found in each cluster's output, it can be inferred that cluster 6 is 
somewhat more associated with the Iraq war.

Q) In 2004, one of the candidates for the Democratic nomination for the President of the United States was Howard Dean, 
John Kerry was the candidate who won the democratic nomination, and John Edwards with the running mate of John Kerry 
(the Vice President nominee). Given this information, which cluster best corresponds to the democratic party?
A) Cluster 4 contains the highest occurrence of terms such as 'dean,' 'kerry,' 'candidate,' 'democrat,' and 'edward,' 
suggesting a stronger connection to the Democratic party.

'''

kmeans = KMeans(n_clusters=7, random_state=1000, n_init=150).fit(df)
clusters = kmeans.predict(df)
clusters = clusters+1

print(np.unique(clusters))


# Create a new DataFrame with the cluster assignments 
df_clusters2 = pd.DataFrame({'kmeans_Cluster': clusters}) 
df_clusters2 = pd.concat([df_clusters2, df], axis=1)  

print(df_clusters2.head(3))

kmeans_counts = df_clusters2['kmeans_Cluster'].value_counts()
print(kmeans_counts)

'''
Q) How many observations are in cluster 3?
A) There are 255 observations in the cluster 3(output from below output)
'''

print('Number of observation in cluster 3 are: ', sum(df_clusters2['kmeans_Cluster'] == 3))

'''
Q) Which cluster has the most observations?
A) Cluster 7 has most observations from below.
'''

max_cluster = kmeans_counts.idxmax()
print('Most observations are in cluster: ', max_cluster)

'''
Q) Which cluster has fewest observations?
A) Cluster 5 has less observations from below.
'''

min_cluster = kmeans_counts.idxmin()
print('Fewest observations are in cluster: ', min_cluster)


# Displaying 6 most frequent words in each cluster

var_names = [f"cluster_{i}" for i in range(1, 8)]
top_words_df = pd.DataFrame(columns=var_names)
for i in range(1, 8):
    cluster = df_clusters2[clusters == i]
    mean_freq = cluster.mean()
    top_words = mean_freq.sort_values(ascending=False)[:6]
    print('Top words in K-means Cluster: ',i,' are: \n',top_words ) 
    top_words_df[var_names[i-1]] = top_words.index.tolist()

print(top_words_df)

'''

Q) Which k-means cluster best corresponds to the Iraq War?
A) Cluster 3

Q) Which k-means cluster best corresponds to the democratic party?
A) Cluster 2

'''

# Crosstab
 
# Create a DataFrame with the cluster assignments for each method
df = pd.DataFrame({ 'HC Clusters': hc_clusters, 'K-Means Clusters': df_clusters2['kmeans_Cluster']})
# Create the cross-tabulation table 
print(pd.crosstab(df['HC Clusters'], df['K-Means Clusters']))

'''
Q) Which Hierarchical Cluster best corresponds to K-Means Cluster 2?

A) Hierarchial Cluster 3 correspond to K-Means cluster 2 because they share the highest number of data points 94 according to the output.

'''

'''
Q) Which Hierarchical Cluster best corresponds to K-Means Cluster 3?

Hierarchial Cluster 6 correspond to K-Means cluster 2 because they share the highest number of data points 174 according to the output.

'''
