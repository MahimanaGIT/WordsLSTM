import pandas as pd
df= pd.read_csv('dataset/Tweets.csv', sep=',')
df.head(10)

#select relavant columns
tweet_df = df[['text','airline_sentiment']]
tweet_df = tweet_df[tweet_df['airline_sentiment'] != 'neutral']
# convert airline_seentiment to numeric
sentiment_label = tweet_df.airline_sentiment.factorize()
# print(sentiment_label)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
tweet = tweet_df.text.values
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(tweet)
vocab_size = len(tokenizer.word_index) + 1
encoded_docs = tokenizer.texts_to_sequences(tweet)
padded_sequence = pad_sequences(encoded_docs, maxlen=200)

from tensorflow import keras

model = keras.models.load_model('./training/model.pb')

test_word ="not satisfactory service"
tw = tokenizer.texts_to_sequences([test_word])
tw = pad_sequences(tw,maxlen=200)
prediction = int(model.predict(tw).round().item())
print(sentiment_label[1][prediction])