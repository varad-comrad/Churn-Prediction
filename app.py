from flask import Flask, render_template, request, url_for, redirect
import subprocess
import pathlib
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__, template_folder='templates')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/update', methods=['GET', 'POST'])
def update():
    complement = []
    if request.method == 'POST':
        complement.append([request.form['Idade'], request.form['UsoMensal'],
                           request.form['Plano'], request.form['SatisfacaoCliente'], 
                           request.form['TempoContrato'], request.form['ValorMensal'], 
                           int(request.form['Churn'] == 'Sim')]) 
        
        if 'Update' in request.form:
            update_file = pathlib.Path(__file__) / 'ml' / 'update.py'
            subprocess.run(['python', str(update_file), '--rows', complement], shell=True)
            complement = []
            return redirect(url_for('train'))
        else:
            return render_template('update.html', complement=complement)
    else:
        return render_template('update.html', complement=complement)


@app.route('/train')
def train():
    train_file = pathlib.Path(__file__).parent / 'ml' / 'train.py'
    subprocess.run(['python', str(train_file)], shell=True)
    return redirect(url_for('index'))


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    result = None
    columns = ['Idade', 'UsoMensal', 'Plano', 'SatisfacaoCliente', 'TempoContrato', 'ValorMensal']
    if request.method == 'POST':
        model_path = pathlib.Path(__file__).parent / 'model' / 'model.pkl'
        with model_path.open('rb') as f:
            model = joblib.load(f)
        inputs = request.form['Idade'], request.form['UsoMensal'], request.form['Plano'], request.form[
            'SatisfacaoCliente'], request.form['TempoContrato'], request.form['ValorMensal']
        inputs = pd.DataFrame([inputs], columns=columns)
        result = "Sim" if model.predict(inputs)[0] == 1 else 'Não'
    return render_template('predict.html', result=result)


if __name__ == '__main__':
    app.run()
