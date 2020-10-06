# -*- coding: utf-8 -*-
from flask import Flask, redirect, request
from pythainlp import word_tokenize # ทำการเรียกตัวตัดคำ
#from pythainlp.word_vector import * # ทำการเรียก thai2vec
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import Word2VecKeyedVectors
from sklearn.metrics.pairwise import cosine_similarity  # ใช้หาค่าความคล้ายคลึง
import numpy as np
import requests, json, os, time

app = Flask(__name__,
            static_url_path='', 
            static_folder='pythaichatbot/static',
            template_folder='pythaichatbot/templates')

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

def phrases_from_json(jsonurl='export.json'):
    # load from local
    if jsonurl == 'export.json' :
        intents = json.load(open(jsonurl,'r'))['intents']
    else :
         intents = json.loads(requests.get(jsonurl).text)['intents']
    itoid = []
    phrase_arr = []
    for intent in intents:
        for phrase in intent['phrases']:
            itoid.append(phrase['intent_id'])
            phrase_arr.append(phrase['value'])
    return phrase_arr, itoid

def responses_from_json(jsonurl='export.json'):
    # load from local
    if jsonurl == 'export.json' :
        intents = json.load(open(jsonurl,'r'))['intents']
    else :
        intents = json.loads(requests.get(jsonurl).text)['intents']
    itoid = []
    response_arr = []
    for intent in intents:
        for response in intent['responses']:
            itoid.append(response['intent_id'])
            response_arr.append(response['value'])
    return response_arr, itoid

def sentence_vectorizer(ss,dim=300,use_mean=True): # ประกาศฟังก์ชัน sentence_vectorizer
    s = word_tokenize(ss)
    vec = np.zeros((1,dim))
    for word in s:
        if word in model.wv.index2word:
            vec+= model.wv.word_vec(word)
        else: pass
    if use_mean: vec /= len(s)
    return vec
def sentence_similarity(s1,s2):
    return cosine_similarity(sentence_vectorizer(str(s1)),sentence_vectorizer(str(s2)))
#prepare thai2vec.bin
os.system('cat ./pythainlp-data/thai2vec.tar* > ./pythainlp-data/thai2vec.tar')
os.system('tar -xf ./pythainlp-data/thai2vec.tar -C ./pythainlp-data')
time.sleep(10)
#model=get_model() # ดึง model ของ thai2vec มาเก็บไว้ในตัวแปร model
model=KeyedVectors.load_word2vec_format('thai2vec.bin', binary=True)
phrase_arr, pitoid = phrases_from_json()
response_arr, ritoid = responses_from_json()
@app.route("/")
def home():
    return redirect("./index.html", code=302)
@app.route("/set_json_url")
def set_intents_json_url():
    jsonurl = request.args.get('url')
    global  phrase_arr, pitoid, response_arr, ritoid
    phrase_arr, pitoid = phrases_from_json(jsonurl)
    response_arr, ritoid = responses_from_json(jsonurl)
    return redirect("./index.html", code=302)
@app.route("/show")
def show_phases():
    separator = '<br> '
    return separator.join(phrase_arr)

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    maxscore = 0.0
    maxscore_id = 0
    loopid = 0
    for phrase in phrase_arr:
        loopid = loopid + 1
        testscore = sentence_similarity(userText,phrase)
        if testscore > maxscore:
           maxscore = testscore
           maxscore_id = loopid
    if maxscore > 0.5:
        rpos = ritoid.index(pitoid[maxscore_id-1])
        respo = response_arr[rpos] + "|||"+ str(pitoid[maxscore_id-1])
    else:
        respo = "ขออภัย ฉันไม่เข้าใจคุณ|||0"
    return respo

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
