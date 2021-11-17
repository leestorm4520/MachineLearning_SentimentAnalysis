# handle imports
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# load the reduced fasttext cluster data
fasttext_df = pd.read_csv('_resources/clusters/fasttext_reduced_clusters.csv')

# plot the fasttext clusters
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="clusters",
    palette=sns.color_palette("hls", 3),
    data=fasttext_df,
    legend="full",
    alpha=0.3
)

# give the fasttext subplot a title
plt.title('FastText KMeans Clusters')

# adjust view
plt.xlim([-20, 20])
plt.ylim([-20, 20])

# save fasttext figure
plt.savefig('_resources/clusters/kmeans_clusters.png')