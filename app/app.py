from flask import Flask, render_template, request
import tensorflow as tf
from src.models.preprocess_data import TextCleanTransformer


app = Flask(__name__)
text_cleaner = TextCleanTransformer()
model = tf.keras.models.load_model('../models/my_model')

prediction_mapping = {
    1: "Disaster!!!",
    0: "There's nothing to be worried about"
}

texts_sent = []


@app.route("/", methods=["POST", "GET"])
def main():
    text = None
    prediction_text = None
    # Checking request type
    if request.method == "POST":
        # text processing
        text = request.form.get('tweet')
        clean_text = text_cleaner.preprocess_sentence(text)
        prediction = round(model.predict([clean_text])[0][0])
        prediction_text = prediction_mapping.get(prediction)

        texts_sent.append({
            'text': text,
            'prediction': prediction
        })

    return render_template('main.html',
                           text=text,
                           prediction_text=prediction_text,
                           texts_sent=texts_sent)


@app.errorhandler(404)
def page_not_found(error):
    return 'page not found error', 404


if __name__ == "__main__":
    app.run(debug=True)