
#from sys import ps1
import pandas as pd
import random
from flask import Flask,render_template,request,jsonify
import pickle
import model 
app=Flask(__name__)
#model=pickle.load(open('/Users/rashisharma/Downloads/deployment/md.pkl','rb'))


@app.route("/",methods=['GET', 'POST'])
def g1():
    name = request.form.get("username")
    name = str(name)
    return render_template('index.html',name=name)
  
@app.route('/Submit', methods=['POST',"GET"])
def g2():
    
    
    name = request.form['username']
    print('name',name)
    mode = model.mod(name)
    
    p1=mode[0]
    p2=mode[1]
    p3=mode[2]
    p4=mode[3]
    p5=mode[4]
    
    return render_template('output.html', name=name,p1=p1,p2=p2,p3=p3,p4=p4,p5=p5)  
  
app.run(debug=True)  
    