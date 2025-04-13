import streamlit as st
import pandas as pd
import numpy as np
import io
from subprocess import Popen, PIPE

import tensorflow as tf
# from IPython.display import display, HTML
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

class The_Neural_Net:
    def __init__(self):
        self.max_len = 0

def load_from_file(self,num_of_epochs):
    # self.num_of_epochs = num_of_epochs
    with open('X_tokenizer.pkl', 'rb') as file:
        self.X_tokenizer = pickle.load(file)
        print("Tokenizer loaded from X_tokenizer.pkl")

    with open('y_tokenizer.pkl', 'rb') as file:
        self.y_tokenizer = pickle.load(file)
        print("Tokenizer loaded from y_tokenizer.pkl")

    model_save_path = f"ner_model_{num_of_epochs}.keras"
    self.model = load_model(model_save_path)
    print(f"Model loaded from {model_save_path}")

The_Neural_Net.load_from_file = load_from_file

def predict(self,model,sentence):
    sentence_tokens = self.X_tokenizer.texts_to_sequences([sentence])

    predictions = model.predict(pad_sequences(sentence_tokens,
                                            maxlen=self.max_len,
                                            padding="post"))
    # print(predictions)
    prediction_ner = np.argmax(predictions,axis=-1)
    # print(prediction_ner)

    NER_tags = [self.y_tokenizer.index_word[num] for num in list(prediction_ner.flatten())]
    final_pred = {"Word":[],"Tag":[]}
    sentence_split = sentence.split()
    for Word,Tag in zip(sentence_split,NER_tags):
        # final_pred[tokens_to_words[i]] = NER_tags[i]
        final_pred["Word"].append(Word)
        final_pred["Tag"].append(Tag)
    return pd.DataFrame(final_pred)
The_Neural_Net.predict = predict
#git commit -m '$(date +"%Y%m%d_%H%M%S") streamlit pkl stuff'

def string_to_dataframe(input_string):
    """Converts a string to a pandas DataFrame."""
    p = Popen("ls -ltrh", stdout=PIPE, stderr=PIPE,shell =True)
    out, err = p.communicate()
    return out.decode()

    try:
        NN_obj = The_Neural_Net()
        num_of_epochs=5
        NN_obj.load_from_file(num_of_epochs=num_of_epochs)
        # sentence = """Is this the real life? Is this just fantasy? Caught in a landslide, no escape from reality"""
        sentence = input_string
        prediction_df = NN_obj.predict(model=NN_obj.model,sentence=sentence)
        return prediction_df

    except Exception as e:
        st.error(f"Error converting string to DataFrame: {e}")
        return None

st.title("String to DataFrame Converter")

input_string = st.text_area("Enter your string (e.g., CSV format):", 
                           value="col1,col2\n1,a\n2,b\n3,c", 
                           height=200)  # Example CSV string

if st.button("Convert to DataFrame"):
    if input_string:
        df = string_to_dataframe(input_string)
        if df is not None:
            st.write("DataFrame:")
            st.dataframe(df)
    else:
        st.warning("Please enter a string.")