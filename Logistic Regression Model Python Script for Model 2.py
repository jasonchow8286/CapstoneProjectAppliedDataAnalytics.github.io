import pandas                  as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn import metrics


PATH     = "C:\\datasets\\"                # Windows
CSV_DATA = "diabetes.csv"
df       = pd.read_csv(PATH + CSV_DATA, sep=',')

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print(df.describe())

print("Value counts: " + str(df['Diabetes_binary'].value_counts()))
le = preprocessing.LabelEncoder()

df['y'] = le.fit_transform(df['Diabetes_binary'])
print(df.tail())
y = df[['y']]


predictorVariables = list(df.keys())
predictorVariables.remove('Diabetes_binary')
predictorVariables.remove('y')

X = df[predictorVariables]
X = X.copy()

y = df['y']


X['bmiBin']  = pd.cut(x=X['BMI'], bins=[0, 18.49, 24.9, 29.9, 34.9, 39.9, 99])

tempDf  = X[['bmiBin']]                 # Isolate columns
dummyDf = pd.get_dummies(tempDf, columns=['bmiBin'])
X      = pd.concat(([X, dummyDf]), axis=1)     # Join dummy df with original
del X['bmiBin']

X.rename(columns={'bmiBin_(0.0, 18.49]':'BMI - Underweight','bmiBin_(18.49, 24.9]':'BMI - Normal', 'bmiBin_(24.9, 29.9]':'BMI - Overweight', 'bmiBin_(29.9, 34.9]':'BMI - Obese Class 1', 'bmiBin_(34.9, 39.9]':'BMI - Obese Class 2', 'bmiBin_(39.9, 99.0]':'BMI - Obese Class 3' }, inplace=True)

print(X.head())


from sklearn.preprocessing import MinMaxScaler
sc_x    = MinMaxScaler()
X_Scale = sc_x.fit_transform(X)


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

test      = SelectKBest(score_func=chi2, k=26)
chiScores = test.fit(X, y) # Summarize scores
np.set_printoptions(precision=3)

print("\nPredictor variables: " + str(list(X.keys())))
print("Predictor Chi-Square Scores: " + str(chiScores.scores_))


cols = chiScores.get_support(indices=True)
print("\nSignificant columns after chi-square test")
print(cols)
features = X.columns[cols]
print("Significant column names after chi-square test")
print(np.array(features))


from   sklearn.model_selection import train_test_split
from   sklearn.linear_model    import LogisticRegression

X = X[['HighBP', 'HighChol', 'Stroke',
 'HeartDiseaseorAttack', 'PhysHlth', 'DiffWalk']]

X_Scale = sc_x.fit_transform(X)

# Split data.
X_train,X_test,y_train,y_test = train_test_split(X_Scale, y, test_size=0.25,
                                                 random_state=0)

# Build logistic regression model and make predictions.
logisticModel = LogisticRegression(fit_intercept=True, solver='liblinear',
                                   random_state=0)
logisticModel.fit(X_train,y_train)
y_pred=logisticModel.predict(X_test)
print("\nPredictions from logistic model")
print(y_pred)


# enumerate splits - returns train and test arrays of indexes.
# scikit-learn k-fold cross-validation
from sklearn.model_selection import KFold
#kfold = KFold(3, True)
kfold = KFold(n_splits=8, shuffle=True)
count = 0

accuracyList = []
precisionList = []
recallList = []
f1List = []

for train_index, test_index in kfold.split(X_Scale):

    X_train, X_test = X_Scale[train_index], X_Scale[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Perform logistic regression.
    logisticModel = LogisticRegression(fit_intercept=True,
                                       solver='liblinear')
    # Fit the model.
    logisticModel.fit(X_train, np.ravel(y_train))

    y_pred = logisticModel.predict(X_test)
    y_prob = logisticModel.predict_proba(X_test)

    # Show confusion matrix and accuracy scores.
    cm = pd.crosstab(y_test, y_pred,
                     rownames=['Actual'],
                     colnames=['Predicted'])
    count += 1
    print("\n***K-fold: " + str(count))

    # Calculate accuracy and precision scores and add to the list.
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)

    print('\nAccuracy: ', accuracy)
    accuracyList.append(accuracy)
    print("\nPrecision: ", precision)
    precisionList.append(precision)
    print("\nRecall: ", recall)
    recallList.append(recall)
    print("\nF1: ", f1)
    f1List.append(f1)
    print("\nConfusion Matrix")
    print(cm)

print("\nAccuracy and Standard Deviation For All Folds:")
print("*********************************************")
print("Average Accuracy: ")
print(np.mean(accuracyList))
print("Accuracy SD: ")
print(np.std(accuracyList))
print("Average Precision: ")
print(np.mean(precisionList))
print("Precision SD: ")
print(np.std(precisionList))
print("Average Recall: ")
print(np.mean(recallList))
print("Recall SD: ")
print(np.std(recallList))
print("Average F1: ")
print(np.mean(f1List))
print("F1 SD: ")
print(np.std(f1List))


from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


# split into train-test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=27)

# train models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# logistic regression
model1 = LogisticRegression()
# knn
model2 = KNeighborsClassifier(n_neighbors=4)

# fit model
model1.fit(X_train, y_train)
model2.fit(X_train, y_train)

# predict probabilities
pred_prob1 = model1.predict_proba(X_test)
pred_prob2 = model2.predict_proba(X_test)

from sklearn.metrics import roc_curve

# roc curve for models
fpr1, tpr1, thresh1 = roc_curve(y_test, pred_prob1[:,1], pos_label=1)
fpr2, tpr2, thresh2 = roc_curve(y_test, pred_prob2[:,1], pos_label=1)

# roc curve for tpr = fpr
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)

# Compute AUC score.
from sklearn.metrics import roc_auc_score

# auc scores
auc_score1 = roc_auc_score(y_test, pred_prob1[:,1])
auc_score2 = roc_auc_score(y_test, pred_prob2[:,1])

print("AUC scores")
print(auc_score1, auc_score2)

# matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn')

# plot roc curves
plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='Logistic Regression')
plt.plot(fpr2, tpr2, linestyle='--',color='green', label='KNN')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')

plt.legend(loc='best')
plt.savefig('ROC',dpi=300)
plt.show();
