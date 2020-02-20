import pickle

from flask import Flask, render_template, redirect, request
from tensorflow.keras.models import load_model

from model_preprocess import pre_process


model = load_model('./my_model.h5')
with open('./tokenizer.pickle', 'rb') as pkl:
    tokenizer = pickle.load(pkl)

app = Flask(__name__, static_folder='.',
            template_folder='.', static_url_path='')


@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/predict')
def predict():
    text = request.args.get('text', '')
    inp = pre_process(text)
    prob = model.predict(inp)[0][0]
    threshold = 0.5
    pred = prob >= threshold and True or False

    return {"prob": str(prob), "threshold": str(threshold), "pred": pred }


if __name__ == '__main__':
    app.run(port=88, debug=True, use_reloader=True)
