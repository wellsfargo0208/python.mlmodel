from flask import Flask, render_template, request
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import re

app = Flask(__name__, template_folder='templates')

model =  tf.keras.models.load_model('sentence_completion.h5')
with open("tokenizer.pkl", 'rb') as file:
    tokenizer = pickle.load(file)


# Text Cleaning
def clean_text(text):
    # removing special characters like @, #, $, etc
    pattern = re.compile('[^a-zA-z0-9\s]')
    text = re.sub(pattern,'',text)
    # removing digits
    pattern = re.compile('\d+')
    text = re.sub(pattern,'',text)
    # converting text to lower case
    text = text.lower()
    return text


def generate_text(model, text, new_words):
    text = clean_text(text)
    new_words = int(new_words)
    for _ in range(new_words):
        text_sequences = np.array(tokenizer.texts_to_sequences([text]))
        testing = pad_sequences(text_sequences, maxlen = 53, padding='pre')
        y_pred_test = np.argmax(model.predict(testing, verbose=0))
        predicted_word = ''
        for word, index in tokenizer.word_index.items():
            if index == y_pred_test:
                predicted_word = word
                break
        text += " " + predicted_word
        
    return text


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['GET', 'POST'])
def generate():

    # If a form is submitted
    if request.method == "POST":
        
        # Get values through input bars
        text = request.form.get("Text")
        no_of_words = request.form.get("NoOfWords")
    
        # Get prediction
        generated_text = generate_text(model, text, no_of_words)
        
    else:
        generated_text = ""
        
    return render_template("generate.html", output = generated_text)


# Running the app
if __name__ == "__main__":
    app.run(debug=True)