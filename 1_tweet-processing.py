# handle imports
import pandas as pd
from TweetProcessor import TweetProcessor

# read in raw tweets from csv
raw_tweets = pd.read_csv("_resources/tweets/raw_tweets.csv")

# clean all the tweets (computationally expensive)
clean_tweets = raw_tweets.apply(lambda row: TweetProcessor.cleanTweet(row['content']), axis=1, result_type='broadcast')

# save the clean tweets
clean_tweets.to_csv('_resources/tweets/clean_tweets.csv', index=False)