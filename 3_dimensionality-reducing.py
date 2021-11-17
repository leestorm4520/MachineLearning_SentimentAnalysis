# handle imports
from gensim.models import FastText

from joblib import load

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# load the fasttext model
fasttext_model = FastText.load('_resources/models/fasttext.model')

# load the kmeans model
km_model_fasttext = load('_resources/models/km_fasttext.model')

# pretty variables for the clusters
cluster_labels = ['neutral_cluster', 'negative_cluster', 'positive_cluster']

# define a function for getting cluster labels
get_cluster_label = np.vectorize(lambda cluster: cluster_labels[cluster])

# get cluster labels for the word2vec kmeans model
fasttext_clusters = get_cluster_label(km_model_fasttext.predict(fasttext_model.wv.vectors))
fasttext_vectors = fasttext_model.wv.vectors

# prepare fasttext dataframe for simple plotting
fasttext_df = pd.DataFrame(fasttext_vectors)
fasttext_df['clusters'] = fasttext_clusters

# perform PCA dimensionality reduction on fasttext
pca = PCA(n_components=3)
fasttext_pca_result = pca.fit_transform(fasttext_vectors)

# perform TSNE dimensionality reduction on fasttext
tsne = TSNE(n_components=2, perplexity=50, n_iter=300)
fasttext_tsne_results = tsne.fit_transform(fasttext_pca_result)

# store fasttext PCA results in the dataframe
fasttext_df['tsne-2d-one'] = fasttext_tsne_results[:,0]
fasttext_df['tsne-2d-two'] = fasttext_tsne_results[:,1]

# save the reduced fasttext cluster data
fasttext_df.to_csv('_resources/clusters/fasttext_reduced_clusters.csv', index=False)