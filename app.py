from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import BaggingClassifier
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import cross_val_score


# load the trained model
model = joblib.load('bagging_classifier.joblib')

# load the test data
y_test = np.load('./y_test.npy', allow_pickle=True)
X_test = np.load('./X_test.npy', allow_pickle=True)

app = Flask(__name__)

# define a route for handling HTTP GET and POST requests to the root URL ('/')
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if all form fields are filled out
        if not all(field in request.form and request.form[field] for field in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']):
            message = 'Please fill out all fields before submitting the form.'
            return render_template('index.html', message=message)

        # get the input data from the form
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(
            request.form['temperature']) if request.form['temperature'] else None
        humidity = float(request.form['humidity']
                         ) if request.form['humidity'] else None
        ph = float(request.form['ph']) if request.form['ph'] else None
        rainfall = float(request.form['rainfall']
                         ) if request.form['rainfall'] else None

        # create a DataFrame with the input data
        data = pd.DataFrame({'N': [N], 'P': [P], 'K': [K],
                             'temperature': [temperature], 'humidity': [humidity],
                             'ph': [ph], 'rainfall': [rainfall]})
        
        y_pred = model.predict(X_test)

        # use the model to generate predictions
        prediction = model.predict(data)

        # model accuracy
        # accuracy = accuracy_score(y_test, y_pred)  #this too will work

        accuracy = model.score(X_test, y_test) #true score

        #model class report
        class_report = classification_report(y_pred, y_test)
        # print(class_report)


        return render_template('index.html', prediction=prediction, report=class_report, bg_score=accuracy, scroll=True)

    # if the request is a GET request, just render the template without any data
    message = 'Please fill out the form below to generate a prediction.'
    return render_template('index.html', message=message)


if __name__ == '__main__':
    app.debug = True
    app.run()
