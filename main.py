import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

word_index=imdb.get_word_index()
reverse_word_index={value:key for key,value in word_index.items()}

model=load_model("simple_rnn_imdb.h5")


def decode_review(encoded_review):
  return ' '.join([reverse_word_index.get(i - 3 ,'?') for i in encoded_review])


def preprocess_text(text):
  words=text.lower().split()
  encoded_review=[word_index.get(word,2)+3 for word in words]
  padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
  return padded_review



def predict_sentiment(review):
  preprocessed_input=preprocess_text(review)
  prediction=model.predict(preprocessed_input)
  print(prediction)
  sentiment="Postive" if prediction[0][0] > 0.5 else "Negative"
  return sentiment, prediction[0][0]


# example_review="this movie was not fantastic! the acting was not great and the plot  was bed."
# example_review=" this is new movie and fntastic movie and the good movie the best movie and the good movie the best movieand the good movie the best movie"
# sentiment,score=predict_sentiment(example_review)
# print(f"Sentiment: {sentiment}")
# print(f"Score: {score}")


import streamlit as st
st.title('IMDB movie review sentiment anaysis')
st.write('Enter a movie review to classify it as positive or negative.')

user_input=st.text_area('Movie review')
if st.button('Classify'):
  preprocessed_input=preprocess_text(user_input)
  prediction=model.predict(preprocessed_input)
  sentiment=sentiment="Postive" if prediction[0][0] > 0.5 else "Negative"
  st.write(f'Sentiment: {sentiment}')
  st.write(f'Prediction Score :{prediction[0][0]}')
else:
  st.write("Please inter a movie review")