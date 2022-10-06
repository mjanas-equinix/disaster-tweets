from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy

import tensorflow as tf

from src.models.preprocess_data import TextCleanTransformer

app = Flask(__name__)
text_cleaner = TextCleanTransformer()
model = tf.keras.models.load_model('../models/my_model')

prediction_mapping = {
    1: "Disaster!!!",
    0: "There's nothing to be worried about"
}


@app.route("/")
def main():
    return render_template('main.html')


@app.route("/response", methods=["POST"])
def response():
    # print(request.form.to_dict())
    text = request.form.get('tweet')
    clean_text = text_cleaner.preprocess_sentence(text)
    prediction = model.predict([clean_text])[0][0]
    prediction_text = prediction_mapping.get(round(prediction))
    return render_template('response.html', text=text, prediction_text=prediction_text)


if __name__ == "__main__":
    app.run(debug=True)