import streamlit as st
import pandas as pd
import numpy as np
import io,pickle
from subprocess import Popen, PIPE

import tensorflow as tf
# from IPython.display import display, HTML
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

class The_Neural_Net:
    def __init__(self):
        self.max_len = 89

def load_from_file(self,num_of_epochs):
    # self.num_of_epochs = num_of_epochs
    with open(f'{pwd}/ML/Assign_2/NER/X_tokenizer.pkl', 'rb') as file:
        self.X_tokenizer = pickle.load(file)
        print("Tokenizer loaded from X_tokenizer.pkl")

    with open(f'{pwd}/ML/Assign_2/NER/y_tokenizer.pkl', 'rb') as file:
        self.y_tokenizer = pickle.load(file)
        print("Tokenizer loaded from y_tokenizer.pkl")

    model_save_path = f"{pwd}/ML/Assign_2/NER/ner_model_{num_of_epochs}.keras"
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
    # print(final_pred)
    # st.write("final_pred:")
    # st.write(final_pred)
    return pd.DataFrame(final_pred)
The_Neural_Net.predict = predict
#git add . ; git add* ;git commit -m '$(date +"%Y%m%d_%H%M%S") streamlit pkl stuff'

def string_to_dataframe(input_string):

    # try:

    # sentence = """Is this the real life? Is this just fantasy? Caught in a landslide, no escape from reality"""
    sentence = input_string
    prediction_df = NN_obj.predict(model=NN_obj.model,sentence=sentence)
    # st.write("Prediction DataFrame:")
    # st.write(prediction_df)
    # st.dataframe(prediction_df)
    return prediction_df

    # except Exception as e:
    #     st.error(f"Error converting string to DataFrame: {e}")
    #     return None

st.title("Named Entity Recognition")

input_string = st.text_area("Enter your string :", 
                           value="Is this the real life? Is this just fantasy? Caught in a landslide, no escape from reality", 
                           height=200)  # Example CSV string

p = Popen("pwd", stdout=PIPE, stderr=PIPE,shell =True)
out, err = p.communicate()
pwd = out.decode().strip()

# df = string_to_dataframe(input_string)

NN_obj = The_Neural_Net()
num_of_epochs=50
NN_obj.load_from_file(num_of_epochs=num_of_epochs)
# if df is not None:
#     st.write("NER:")
#     st.dataframe(df)


# st.write(f"pwd {pwd}", out.decode())

# cmd = f"ls -ltrh {pwd}/ML/Assign_2/NER/*"
# st.write(f"cmd:\t{cmd}")
# p = Popen(cmd, stdout=PIPE, stderr=PIPE,shell =True)
# out, err = p.communicate()
# files = out.decode().split("\n")
# for i in files:
#     if i:
#         st.write(i)
# st.write(f"ls -ltrhR\n\n{pwd}", out.decode())


# st.write("model location:", f"{pwd}/ML/Assign_2/NER/X_tokenizer.pkl")
# p = Popen(f"ls {pwd}/ML/Assign_2/NER/*", stdout=PIPE, stderr=PIPE,shell =True)
# out, err = p.communicate()
# ls = out.decode()
# st.write(f"ls {pwd}/ML/Assign_2/NER/*: {ls}")
if st.button("Get NERs"):
    if input_string:
        df = string_to_dataframe(input_string)
        if df is not None:
            st.write("NER:")
            st.dataframe(df)
    else:
        st.warning("Please enter a string.")

# """Converts a string to a pandas DataFrame."""
