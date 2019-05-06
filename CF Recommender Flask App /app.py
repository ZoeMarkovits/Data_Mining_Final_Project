import flask
from flask import request
from flask import render_template
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import coo_matrix

app = flask.Flask(__name__)

with open('cf_df.pkl', 'rb') as a:
    cf_df = pickle.load(a)

with open('ratings_matrix.pkl', 'rb') as b:
    ratings_matrix = pickle.load(b)

with open('svd.pkl', 'rb') as c:
    svd = pickle.load(c)

with open('svd_out.pkl', 'rb') as d:
    svd_out = pickle.load(d)



@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')

def get_similar_beers(beerID, n=6):
    vec = cosine_similarity(svd.components_[:,beerID-1].reshape(1,-1),svd.components_.T).reshape(-1)
    res = vec.argsort()[-n:][::-1]
    return list(map(lambda x: (cf_df.Full_Beername[x+1], np.round(vec[x],3)), res))

get_similar_beers(2)

@app.route('/result',methods = ['POST'])
def result():
    user_beer = flask.request.form['user_beer']
    beer_options = (cf_df.Full_Beername).tolist()
    if user_beer in beer_options:
        id = cf_df[cf_df.Full_Beername == user_beer].index[0]
        sims = get_similar_beers(id, n=10)
        beers = []
        for sim in sims:
            beers.append(sim[0])
        result = beers
        result = str(result)
        result = result[1:-1]
    else:
        result = "Sorry, please try entering another beer!"
    return render_template("result.html",result=result)


if __name__ == '__main__':
    app.run(debug=True)
