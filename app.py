from flask import Flask,render_template,request
import pickle
import numpy as np
import pandas as pd

# Load the model and other pickled files
model = pickle.load(open('XGB.pkl','rb'))
lea = pickle.load(open('lea.pkl','rb'))
ler = pickle.load(open('ler.pkl','rb'))
leg = pickle.load(open('leg.pkl','rb'))
let = pickle.load(open('let.pkl','rb'))
scr = pickle.load(open('scr.pkl','rb'))
sce = pickle.load(open('sce.pkl','rb'))
scs = pickle.load(open('scs.pkl','rb'))


# Create application
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods = ['POST'])
def predict():
    Drug =request.form.get('drugname')
    Gender =request.form.get("gender")
    Race =request.form.get('race')
    Age =request.form.get("age")
    Satisfaction_Rating =float(request.form.get("satisfaction"))
    Effectiveness_Rating =float(request.form.get("effectiveness"))
    Review_drug = request.form.get("review")
    Year = int(request.form.get("year"))
         
          
    # Perform classification prediction
    side_effect = model.predict(np.array([ler.transform([Race])[0], lea.transform([Age])[0], sce.transform([[Effectiveness_Rating]])[0,0], scs.transform([[Satisfaction_Rating]])[0,0], leg.transform([Gender])[0], Year, scr.transform([[Review_drug]])[0,0]]).reshape(1,-1))[0]
    if side_effect == 0:
        side_effect = 'Extremely severe'
    elif side_effect == 1:
        side_effect = 'Mild'
    elif side_effect == 2:
        side_effect = 'Moderate'
    elif side_effect == 3:
        side_effect = 'No Side Effect'
    else:
        side_effect = 'Severe'
   
        
    return render_template('result.html', predicted_text=' Impact of  {} on patient:'.format(Drug), predicted=side_effect, drug=Drug)
#Finally, start the flask server and run our web page locally on the computer by calling app.run() and then enter http://localhost:5000 on the browser.
if __name__ == '__main__':
    #Run the application
    app.run(debug=True)