from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

def home(request):
    return render(request, "home.html")

def predict(request):
    return render(request, "predict.html")

def result(request):
    csv_path = settings.BASE_DIR / 'static' / 'data' / 'USA_Housing.csv'
    data = pd.read_csv(csv_path)
    data = data.drop(['Address'], axis=1)
    X = data.drop('Price', axis=1)
    Y = data['Price']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.30)
    model = LinearRegression()
    model.fit(X_train, Y_train)

    var1 = float(request.GET['n1'])
    var2 = float(request.GET['n2'])
    var3 = float(request.GET['n3'])
    var4 = float(request.GET['n4'])
    var5 = float(request.GET['n5'])

    pred = model.predict(np.array([var1, var2, var3, var4, var5]).reshape(1, -1))
    predicted_price = float(pred[0])

    price = f"The predicted price is ${predicted_price:,.2f}"

    return render(request, "predict.html", {"result2": price})
