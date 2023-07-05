
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import os
class Model:
    df = pd.read_csv("dataset/depressionDataset.csv")
    df.drop(['id', 'score'], axis=1, inplace=True)
    df['q1'] = df['q1'].fillna(df['q1'].mode()[0])
    df['q2'] = df['q2'].fillna(df['q2'].mode()[0])
    df['q3'] = df['q3'].fillna(df['q3'].mode()[0])
    df['q4'] = df['q4'].fillna(df['q4'].mode()[0])
    df['q5'] = df['q5'].fillna(df['q5'].mode()[0])
    df['q6'] = df['q6'].fillna(df['q6'].mode()[0])
    df['q7'] = df['q7'].fillna(df['q7'].mode()[0])
    df['q8'] = df['q8'].fillna(df['q8'].mode()[0])
    df['q9'] = df['q9'].fillna(df['q9'].mode()[0])
    df['q10'] = df['q10'].fillna(df['q10'].mode()[0])
    df['class'] = df['class'].fillna(df['class'].mode()[0])

    X = df.drop(['class'], axis=1)

    y = df['class']

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=24)
    # instantiate classifier with linear kernel and C=1.0
    linear_svc = SVC(kernel='linear', C=1.0)

    # fit classifier to training set
    linear_svc.fit(X_train,y_train)

    predictions = linear_svc.predict(X_test)
    cm = confusion_matrix(y_test, predictions)
    accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
    print(f"Model has accuracy of {accuracy * 100} % ")


