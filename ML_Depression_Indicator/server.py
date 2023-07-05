from flask import Flask,request,render_template,flash,redirect,session,abort,jsonify
from model import Model
import pandas as pd
import os

app = Flask(__name__)

@app.route('/')
def root():
    return render_template('indexNew.html')


@app.route('/predict',methods=['Post'])
def predict():
    q1 = int(request.form['a1'])
    q2 = int(request.form['a2'])
    q3 = int(request.form['a3'])
    q4 = int(request.form['a4'])
    q5 = int(request.form['a5'])
    q6 = int(request.form['a6'])
    q7 = int(request.form['a7'])
    q8 = int(request.form['a8'])
    q9 = int(request.form['a9'])
    q10 = int(request.form['a10'])

    values = [q1, q2, q3, q4, q5, q6, q7, q8, q9, q10]
    model = Model()
    classifier = model.linear_svc
    prediction = classifier.predict([values])
    if prediction[0] == 0:
        result = 'Your Depression test result : No Depression'
        symptom = 'No depression symptom.'
        note = 'You have a good environment and good lifestyle.'
    if prediction[0] == 1:
        result = 'Your Depression test result : Mild Depression'
        symptom = 'You might have these following symptom: feeling of sadness, loss of appetite, sleeping problems, reduced eergy level and also difficulties concentrating.'
        note = 'For your medication, we suggest you to change your lifestyle by doing some exercise, having music therapy, get some balance diet, having contact with other people and change your sleep habit.'
    if prediction[0] == 2:
        result = 'Your Depression test result : Moderate Depression'
        symptom = 'You might have these following symptom: feeling of sadness, loss of appetite, sleeping problems, reduced eergy level and also difficulties concentrating.'
        note = 'For your medication, we suggest you to change your lifestyle by increase your exercise level by doing some recreational activities, having music theraphy, relaxing, having medication (antidepresasant) and having the interacting with pets and animals.'
    if prediction[0] == 3:
        result = 'Your Depression test result : Moderately severe Depression'
        symptom = 'You might have these following symptom: avoiding social activities, changes appetite, having difficulty in concentrating, excessive worry, always fatigue, feeling hopelessness.'
        note = 'For your medication, we suggest you to have Psychotheraphy treatment such as Dialectical behaviour theraphy (DBT), Interpersonal theraphy (IPT) and Psychodynamic theraphy.'
    if prediction[0] == 4:
        result = 'Your Depression test result : Severe Depression'
        symptom = 'You might have these following symptom: Feeling depresed, losing interest, sleep difficulty, always fatigue, feeling hopelessness and suicidal thoughts or behaviours.'
        note = 'For your medication, we suggest you to have Pcychotheraphy treatment and Medicatications treatment.'
    return render_template("resultNew.html", result=result, symptom=symptom, note=note)


app.secret_key = os.urandom(12)
app.run(port=5997, host='0.0.0.0', debug=True)
