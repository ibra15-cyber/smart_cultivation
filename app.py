from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import cross_val_score


# load the trained model
model = joblib.load('bagging_classifier.joblib')

###we want to calculate the score and generate report
df = pd.read_csv('./crop_recommendation.csv')
x = df.drop(['label'], axis=1) # our independent variable; removing the label as part of our data and saving the rest as x
y = df['label']

#splitting data into train and test
x_train,x_test,y_train,y_test = model_selection.train_test_split(x,y)

#piping it
pipe1 = make_pipeline(preprocessing.StandardScaler(),RandomForestClassifier(n_estimators = 10)) #

# BaggingClassifier trains multiple instances of a base estimator (such as a decision tree, k-nearest neighbors, or logistic regression) on different subsets of the training data and aggregates their predictions to make a final prediction.
bag_model = BaggingClassifier(base_estimator=pipe1,n_estimators=100,
                                    oob_score=True,random_state=0,max_samples=0.8)

bag_model.fit(x_train,y_train) #training the module
# create an instance of the Flask class
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

        # use the model to generate predictions
        prediction = model.predict(data)

        bg_score = bag_model.score(x_test, y_test) #bag_model.score(x_test,y_test) would calculate the mean accuracy of the BaggingClassifier model on the test data x_test and y_test

        predicted = bag_model.predict(x_test) #would generate predicted labels for the test data x_test using the BaggingClassifier model that was trained earlier.
        report = classification_report(y_test, predicted)


        return render_template('index.html', prediction=prediction, report=report, bg_score=bg_score, scroll=True)

    # if the request is a GET request, just render the template without any data
    message = 'Please fill out the form below to generate a prediction.'
    return render_template('index.html', message=message)


if __name__ == '__main__':
    app.debug = True
    app.run()
