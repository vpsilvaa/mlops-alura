from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth
from textblob import TextBlob
from googletrans import Translator
from sklearn.linear_model import LinearRegression
import pickle
import os

#pip install -r requirements.txt

model = pickle.load(open('../../models/model.sav', 'rb'))

columns = ['tamanho', 'ano', 'garagem']

app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = os.environ.get('BASIC_AUTH_USERNAME')
app.config['BASIC_AUTH_PASSWORD'] = os.environ.get('BASIC_AUTH_PASSWORD')

basic_auth = BasicAuth(app)
@app.route('/')
def home():
    return '1 api'

@app.route('/sentimento/<frase>')
@basic_auth.required
def sentimento(frase):
    """
    Return the sentiment of a given phrase. The sentiment is a float within the
    range [-1.0 to 1.0] where -1.0 is very bad and 1.0 is very good.

    Parameters
    ----------
    frase : str
        The phrase to be analyzed.

    Returns
    -------
    str
        The sentiment of the phrase.
    """

    translator = Translator()
    translated = translator.translate(frase, src='pt', dest='en')
    tb = TextBlob(translated.text)
    polaridade = tb.sentiment.polarity
    return "polaridade: {}".format(polaridade)

@app.route('/preco/', methods=['POST'])
@basic_auth.required
def preco():
    """
    Return the price of a house given its attributes.

    Parameters
    ----------
    dados : dict
        A dictionary with the following keys:
        - tamanho: the size of the house
        - ano: the year of the house
        - garagem: the number of parking spaces

    Returns
    -------
    str
        The price of the house.
    """

    dados = request.get_json()
    input = [dados[col] for col in columns]
    valor = model.predict([input])
    return jsonify(preco = valor[0])

app.run(host="0.0.0.0", port=8080, debug=True)

