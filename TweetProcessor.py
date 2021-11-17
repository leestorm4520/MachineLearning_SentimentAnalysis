# handle imports
import unicodedata
from gensim.models import phrases
import pandas as pd

from gensim.models.phrases import ENGLISH_CONNECTOR_WORDS, Phrases

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import re
import regex

import html

import emoji


# define a helper class
class TweetProcessor:

  def __init__(self):
    # read clean tweets in from csv
    self.clean_tweets = pd.read_csv('_resources/tweets/clean_tweets.csv')

    # split all the tweets
    self.split_clean_tweets = self.clean_tweets.applymap(lambda content: str(content).split())

    # prepare a phraser to catch common english phrases
    self.phrase_model = Phrases(self.split_clean_tweets['content'], min_count=1, threshold=1, connector_words=ENGLISH_CONNECTOR_WORDS)

    # phrase the split clean tweets
    self.phrases = self.phrase_model[self.split_clean_tweets['content']]

  # define tweet cleaning function
  @staticmethod
  def cleanTweet(tweet):

    # sub out urls, regex: r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    tweet = regex.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", "URL", tweet)

    # remove apostrophes, quotation marks, and similar (´ ` ˝ ̏  ' " ‘ ’ “ ”) regex: r"[\´\`\˝\ ̏\'\"\‘\’\“\”]*"
    tweet = regex.sub(r"""['´`˝"‘’“”]""", '', tweet)

    # sub html enitity references for unicode equivalents
    tweet = html.unescape(tweet)

    # find decimals (dddd.dddd) -> (dddd_dec_dddd)
    tweet = regex.sub(r"(\d)\.(\d)", r"\1_dec_\2", tweet)

    # find fractions (dddd/dddd) -> (dddd_frac_dddd)
    tweet = regex.sub(r"(\d)\/(\d)", r"\1_frac_\2", tweet)

    # pad punctuation with spaces (except @ # _)
    tweet = regex.sub(r"(\p{P}(?<![#@_]))", r" \1 ", tweet)

    # pad /'s with spaces except for in fractions (digit/digit)
    tweet = regex.sub(r"(\D)(\/)(\D)", r"\1 \2 \3", tweet)

    # replace . . . with …
    tweet = regex.sub(r"\.\s*\.\s*\.", "…", tweet)

    # re insert decimals
    tweet = regex.sub(r"(\d)_dec_(\d)", r"\1\.\2", tweet)

    # re insert fractions
    tweet = regex.sub(r"(\d)_frac_(\d)", r"\1\/\2", tweet)

    # find emojis
    emojis = re.findall(emoji.get_emoji_regexp(), tweet)

    # sub emojis with emoji names and pad
    for match in emojis:
      tweet = tweet.replace(match, " "+ unicodedata.name(match[0]) +" ")

    # remove extra emoji variant codes
    tweet = regex.sub(r"\s\ufe0f\s", ' ', tweet)

    # find user mentions
    users = regex.findall(r"@\S*", tweet)

    # sub out user mentions
    for count, match in enumerate(users):
      tweet = tweet.replace(match, " user"+str(count+1)+" ")

    # find hashtags
    hashtags = regex.findall(r"(#\S*)", tweet)

    # transform to lowercase
    tweet = tweet.lower()

    # remove english stop words
    tweet = ' '.join([word for word in tweet.split() if word not in stopwords.words('english')])

    # lemmatize tweet
    tweet = ' '.join([WordNetLemmatizer().lemmatize(word) for word in tweet.split()])

    # re-sub in the hashtags (#original: as lower) -> (#OrigiNal: as captured)
    for match in hashtags:
      tweet = tweet.replace(match.lower(), match)

    # replace all whitespace with single spaces
    tweet = regex.sub(r"(?<!^)[\s]+(?!$)", ' ', tweet)

    # return the clean tweet
    return tweet