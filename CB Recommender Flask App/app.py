import flask
from flask import request
from flask import render_template
import pickle
import pandas as pd
import numpy as np

app = flask.Flask(__name__)

with open('data.pkl', 'rb') as f:
    recommendations_df = pickle.load(f)

@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')

def cosine_similiarity(x1,x2):
    num=np.dot(x1,x2)
    magx1=np.sqrt(sum([a**2 for a in x1]))
    magx2=np.sqrt(sum([b**2 for b in x2]))
    denom=1
    if magx1 ==0 or magx2 ==0:
        denom ==1
    else:
        denom =np.dot(magx1,magx2)
    return num/denom

@app.route('/result',methods = ['POST'])
def result():
    user_beer = flask.request.form['user_beer']
    beer_options = (recommendations_df.index).tolist()
    if user_beer in beer_options:
        beer_input_values = recommendations_df.loc[user_beer]
        all_similarities=[]
        for i in recommendations_df.index:
            all_similarities.append(cosine_similiarity(recommendations_df.loc[i],beer_input_values))
        beer_similarities = sorted([(v,i) for i,v in enumerate(all_similarities)],reverse=True)[1:6]
        beer_values = [m[1] for m in beer_similarities]
        result = recommendations_df.iloc[beer_values].index.tolist()
        result = str(result)
        result = result[1:-1]
    else:
        result = "Sorry, please try entering another beer!"
    return render_template("result.html",result=result)


if __name__ == '__main__':
    app.run(debug=True)
