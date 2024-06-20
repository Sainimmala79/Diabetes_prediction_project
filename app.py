import numpy as np
from flask import Flask,request,render_template
import pickle

app=Flask(__name__)
scaler=pickle.load(open(r'C:\Users\saini\projects\diabetes\scaler.pkl','rb'))
model=pickle.load(open(r'C:\Users\saini\projects\diabetes\ml_model.pkl','rb'))

#decorater
@app.route("/")
def home():
    return render_template(r"home.html")

@app.route("/predict",methods=['post'])
def predict():
    val_float=[float(x) for x in request.form.values()]
    val_arr=[np.array(val_float)]
    res=model.predict(scaler.transform(val_arr))
    return render_template(r'result.html',prediction=res)

if __name__=="__main__":
    app.run(debug=True)


