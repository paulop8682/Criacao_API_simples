import pickle
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
import seaborn as sns

colunas = ['age', 'gender', 'education_level']
modelo = pickle.load(open("/home/paulo/ridge_model.sav", "rb"))

app = Flask(__name__)
@app.route('/')
def home():
    return "minha primeira api"

@app.route('/salary/', methods=['POST'])
def teste():
    dados = request.get_json()
    dados_input = [dados[col] for col in colunas]
    salario = modelo.predict([dados_input])
    return jsonify(salario=salario[0])

app.run(debug=True)