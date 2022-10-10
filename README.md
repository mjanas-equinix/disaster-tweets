# disaster-tweets
# Repository created for workshop
This repository was created for the workshop to present how flask can be useful in the data science context.

## Problem statement
The repository contains model that predicts if a given text contains information about natural disaster.
Build an app with a GUI with a form containing a text field and a submit button. 
Once user submits the tweet, the page should show a response with the model prediction (disaster / not a disaster).

Example of using text cleaning module and model for predictions is available in src/models/make_predictions.py

## How to start
- clone the repository<br>
- create a virtual environment, activate it and install packages from requirements.txt<br>
``pip install -r requirements.txt``<br>
- install spacy model<br>
``python -m spacy download en_core_web_sm``
- create new directory "app" and start building an app in it!
