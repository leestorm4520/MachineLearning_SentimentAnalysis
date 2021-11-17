# handle imports
import pandas as pd

import multiprocessing
from gensim.models import FastText
from sklearn.cluster import KMeans

from joblib import dump, load

from TweetProcessor import TweetProcessor

# create a TweetProcessor
tweet_processor = TweetProcessor()

# build fasttext model
cpu_count = multiprocessing.cpu_count()

fasttext_model = FastText(min_count=10,
                     window=4,
                     vector_size=50,
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=30,
                     workers=cpu_count-1)

fasttext_model.build_vocab(tweet_processor.phrases)

# build kmeans model
km_model = KMeans(n_clusters=3, max_iter=2000, random_state=True, n_init=50)

# train the fasttext model
fasttext_model.train(tweet_processor.phrases, total_examples=fasttext_model.corpus_count, epochs=30)

# train a kmeans model for fasttext vectorization strategy
km_model_fasttext = km_model.fit(X=fasttext_model.wv.vectors)

# save the fasttext model
fasttext_model.save('_resources/models/fasttext.model')

# save the kmeans model
dump(km_model_fasttext, '_resources/models/km_fasttext.model')

# load the fasttext model
fasttext_model = FastText.load('_resources/models/fasttext.model')

# load the kmeans model
km_model_fasttext = load('_resources/models/km_fasttext.model')