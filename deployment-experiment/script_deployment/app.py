import pickle
from flask import Flask, request
import pandas as pd

app = Flask(__name__)

with open("model/model_numpy.pkl", "rb") as model_file:
    model_numpy = pickle.load(model_file)

with open("model/model_pandas.pkl", "rb") as model_file:
    model_pandas = pickle.load(model_file)

FEATURE = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
LABEL = ['Iris setosa', 'Iris versicolor', 'Iris virginica']
@app.route('/')
def index():
    return {"status":"SUCCESS",
            "message": "Service is Up"}, 200

@app.route('/sapa')
def sapa_nama():
    args = request.args
    nama =  args.get('nama', default='Vincen')
    jobs = args.get('jobtitle', default='ML Engineering')
    return {"status":"SUCCESS",
            "message":f"Hallo {nama}, Pekerjaan anda adalah {jobs}"}, 200

@app.route('/predict/numpy')
def predict_numpy():
    args = request.args
    sl = args.get('s1', default=0.0, type=float)
    sw = args.get('sw', default=0.0, type=float)
    pl = args.get('pl', default=0.0, type=float)
    pw = args.get('pw', default=0.0, type=float)
    new_data = [[sl ,sw, pl, pw]]
    res = model_numpy.predict(new_data)
    res = LABEL[res[0]]
    return {"status":"SUCCESS",
            "input type:": "Numpy array",
            "input" : {
                'sepal length':sl,
                'Sepal width' :sw,
                'petal length':pl,
                'petal width' :pw
            },
            "result": res}, 200

@app.route('/predict/pandas')
def predict_pandas():
    args = request.args
    sl = args.get('s1', default=0.0, type=float)
    sw = args.get('sw', default=0.0, type=float)
    pl = args.get('pl', default=0.0, type=float)
    pw = args.get('pw', default=0.0, type=float)
    new_data = [[sl ,sw, pl, pw]]
    new_data = pd.DataFrame(new_data, columns=FEATURE)
    res = model_pandas.predict(new_data)
    res = LABEL[res[0]]
    return {"status":"SUCCESS",
            "input type:": "Pandas dataFrame",
            "input" : {
                'sepal length':sl,
                'Sepal width' :sw,
                'petal length':pl,
                'petal width' :pw
            },
            "result": res}, 200
app.run(debug=True)