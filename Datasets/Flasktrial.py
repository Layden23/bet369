import pandas as pd,numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score,cohen_kappa_score,confusion_matrix,recall_score,precision_score
from sklearn.linear_model import LogisticRegression
import plotly.express as px
from datetime import datetime
import time, random
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier


train = pd.read_csv("C:/Users/andre/Google Drive/Github/bet369/Datasets/train3.csv", header = 0)

len(train["MatchID"].unique())

train = train.drop(train[train["MatchID"] == "MatchID"].index)

#Remove Strange Matches
train = train.loc[(train['HostPossessionFT'] != "0") & (train['GuestPossessionFT'] != "0")]
#Fix Goal Problem
train["Goal"] = pd.to_numeric(train["HostGoalsFT"]) + pd.to_numeric(train["GuestGoalsFT"])

train[train["Status"] == "HT"].iloc[:,-20:].sort_values(by = "Goal", ascending = False)

s = 203

#Remove columns with lots of missing values
del train["BettingCuoteHT"]
del train["BettingCuoteFT"]
del train["HostGoalsFT"]
del train["GuestGoalsFT"]

#Creating train with matches that score in the first half
# train.Status = train.Status.replace(["HT","FT"],[45,90])
train = train.drop(train[train["Status"] == "NS"].index)
train = train.drop(train[train["Status"] == "HT"].index)
train = train.drop(train[train["Status"] == "FT"].index)
#Transform variables to numeric
for i in train.iloc[:,12:].columns:
    train[i] = pd.to_numeric(train[i])
#Creating train with matches that don't score in the first half
train0 = train[(train["Goal"] == 0) & (train["Status"] <= 45)]
random.seed(69)
train0 = train0.groupby('MatchID').apply(lambda x: x.sample(1))
train00 = train[(train["Goal"] == 0) & (train["Status"] <= 45)]
train00 = train00.groupby('MatchID').apply(lambda x: x.sample(1))
train000 = train[(train["Goal"] == 0) & (train["Status"] <= 45)]
train000 = train000.groupby('MatchID').apply(lambda x: x.sample(1))

train1 = train[(train['Status'] <= 45) & (train["Goal"]==1)]
train1 = train1.sort_values("Status").groupby("MatchID", as_index=False).first()

train1.sort_values("Goal", ascending = False).iloc[:,-20:]

ntrain = train1.append(train0)
ntrain1 = ntrain.append(train00)
ntrain2 = ntrain.append(train000)
ntrain1 = ntrain1.drop_duplicates(subset="MatchID",keep='first')
ntrain2 = ntrain1.drop_duplicates(subset="MatchID",keep='first')
ntrain1 = ntrain1.dropna()
ntrain2 = ntrain2.dropna()

ntrain = ntrain.dropna()
ntrain = ntrain.drop_duplicates(subset="MatchID",keep='first')

#Oversampling
ntrain = ntrain.append(ntrain1[ntrain1["Goal"]==0])
ntrain = ntrain.append(ntrain2[ntrain2["Goal"]==0])

# Indicies of each class' observations
random.seed(s)
i_class0 = np.where(ntrain["Goal"] == 1)[0]
i_class1 = np.where(ntrain["Goal"] == 0)[0]

#Get size of the underrepresented class
n_class0 = len(i_class0)

i_class1_downsampled = np.random.choice(i_class1, size=n_class0, replace=False)
ntrain = ntrain.iloc[i_class0,:].append(ntrain.iloc[i_class1_downsampled,:])

ntrain.sort_values(by = "Goal", ascending = False)

# =============================================================================
# Feature engineering
# =============================================================================

X = ntrain.iloc[:,12:33]
# stand = MinMaxScaler().fit(X)
# stdX = pd.DataFrame(stand.transform(X))
# pca = PCA(0.99)
# pca.fit(stdX)
# pcX = pca.transform(stdX)
y = ntrain["Goal"]
#Transform variables to numeric
for i in X.columns:
    X[i] = pd.to_numeric(X[i])
y = pd.to_numeric(y)

random.seed(s)
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size = 0.7)
                                                    
#Random Forest
#random.seed(s)
#Random Forest
rf = RandomForestClassifier(n_estimators = 200)
rf.fit(X_train, y_train)

# Logistic Regression
random.seed(s)
lr = LogisticRegression()
lr.fit(X_train, y_train)

#XGBTree 
random.seed(s)
# cv_params = {'max_depth': [3,5,7], 'min_child_weight': [1,3,5]}
# ind_params = {'learning_rate': 0.1, 'n_estimators': 1000, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8, 
#              'objective': 'binary:logistic'}
# xgb = GridSearchCV(xgb.XGBClassifier(**ind_params), 
#                             cv_params, 
#                              scoring = 'accuracy', cv = 5, n_jobs = -1)
xgb = xgb.XGBClassifier()
xgb.fit(X_train,y_train)

#Predictions
rfpreds = rf.predict(X_test)
lrpreds = lr.predict(X_test)
xgbpreds = xgb.predict(X_test)

#Probabilities
rfprobs = rf.predict_proba(X_test)[:,1]
lrprobs = lr.predict_proba(X_test)[:,1]
xgbprobs = xgb.predict_proba(X_test)[:,1]

#Raül probs
Vscore2 = X_test['HostCornersFT']*0.9+X_test['GuestCornersFT']*0.9+X_test['HostYCFT']+X_test['GuestYCFT'] 
+ X_test['HostRCFT']*-2.0+X_test['GuestRCFT']*-2.0+X_test['HostonTargetFT']*9.0+X_test['GuestonTargetFT']*9.0 
+ X_test['HostoffTargetFT']*5.0+X_test['GuestoffTargetFT']*5.0+X_test['HostAttacksFT']*0.7 
+ X_test['GuestAttacksFT']*0.7+X_test['HostDAttacksFT']+X_test['GuestDAttacksFT']
Vscore2
Vscore2 = Vscore2/X_test['Status']
raulprobs = Vscore2 * 10
raulpreds = round(raulprobs)

X_test["raulprobs"] = raulprobs
X_test.iloc[:,-20:]

results = pd.DataFrame({"Goal": y_test,"rfpreds" : rfpreds,"lrpreds": lrpreds, "xgbpreds": xgbpreds,
              "rfprobs": rfprobs, "lrprobs": lrprobs, "xgbprobs": xgbprobs, "raulpreds": raulpreds, "raulprobs": raulprobs})

rf_test = results[results["rfprobs"] >= 0.8]
lr_test = results[results["rfprobs"] >= 0.8]
xgb_test = results[results["raulprobs"] >= 0.8]
#n_test = results[results["nprobs"] >= 0.8]
raul_test = results[results["raulprobs"] >= 0.8]
pd.DataFrame({"":["0.9 Accuracy","Accuracy","% 0.9 preds","% 0.8 preds","% 0.7 preds"],
              "rf": [accuracy_score(rf_test["Goal"],rf_test["rfpreds"]), 
                                 accuracy_score(rfpreds,y_test), 
                     len(results[results["rfprobs"] >= 0.9])*100/len(X_test),
                    len(results[results["rfprobs"] >= 0.8])*100/len(X_test),
                    len(results[results["rfprobs"] >= 0.7])*100/len(X_test)],
              "lr": [accuracy_score(lr_test["Goal"],lr_test["lrpreds"]), 
                                 accuracy_score(lrpreds,y_test), 
                     len(results[results["lrprobs"] >= 0.9])*100/len(X_test),
                    len(results[results["lrprobs"] >= 0.8])*100/len(X_test),
                    len(results[results["lrprobs"] >= 0.7])*100/len(X_test)],
              "xgb": [accuracy_score(xgb_test["Goal"],xgb_test["xgbpreds"]), 
                                 accuracy_score(xgbpreds,y_test),
                      len(results[results["xgbprobs"] >= 0.9])*100/len(X_test),
                     len(results[results["xgbprobs"] >= 0.8])*100/len(X_test),
                     len(results[results["xgbprobs"] >= 0.7])*100/len(X_test)],
              "raul": [accuracy_score(raul_test["Goal"],raul_test["raulpreds"]), 
                                 accuracy_score(raulpreds,y_test), 
                       len(results[results["raulprobs"] >= 0.9])*100/len(X_test),
                      len(results[results["raulprobs"] >= 0.8])*100/len(X_test),
                      len(results[results["raulprobs"] >= 0.7])*100/len(X_test)]
               })
    
print(accuracy_score(results[results["rfprobs"] >= 0.7]["Goal"],results[results["rfprobs"] >= 0.7]["rfpreds"]))

confusion_matrix(xgbpreds,y_test)

confusion_matrix(lrpreds,y_test)

confusion_matrix(rfpreds,y_test)

confusion_matrix(raulpreds,y_test)

print("Metric   "," RF "," LR ","XGB")
print("Accuracy ", round(accuracy_score(rfpreds,y_test),2), round(accuracy_score(lrpreds,y_test),2), round(accuracy_score(xgbpreds,y_test),2))
print("Kappa    ",round(cohen_kappa_score(rfpreds,y_test),2),round(cohen_kappa_score(lrpreds,y_test),2),round(cohen_kappa_score(xgbpreds,y_test),2))
print("Precision",round(precision_score(rfpreds,y_test),2),round(precision_score(lrpreds,y_test),2),round(precision_score(xgbpreds,y_test),2))
print("Recall   ", round(recall_score(rfpreds,y_test),2),round(recall_score(lrpreds,y_test),2),round(recall_score(xgbpreds,y_test),2))

import os
os.chdir('C:/Users/andre/Desktop/Ubiqum/Final Project/Datasets/CSVs/'+datetime.now().strftime('%Y-%m-%d')+'/')
os.getcwd()

dirname = ('C:/Users/andre/Desktop/Ubiqum/Final Project/Datasets/CSVs/'+datetime.now().strftime('%Y-%m-%d')+'/')
filename = datetime.now().strftime('gambling-%Y-%m-%d-%H-%M.csv')
test = pd.read_csv(dirname+filename, header = 0)

del test["BettingCuoteHT"]
del test["BettingCuoteFT"]
del test["HostGoalsFT"]
del test["GuestGoalsFT"]
test = test.dropna()
test = test.loc[(test['HostPossessionFT'] != 0) & (test['GuestPossessionFT'] != 0)]
test = test.drop(test[test["Status"] == "HT"].index)
test = test.drop(test[test["Status"] == "NS"].index)
test = test.drop(test[test["Status"] == "FT"].index)
ntest = test.iloc[:,12:33]
ntest["Status"] = pd.to_numeric(ntest["Status"])
# pctest = pca.transform(ntest)
tpreds = rf.predict(ntest)
tprobs = rf.predict_proba(ntest)[:,1]
test["Prediction"] = tpreds
test["Probability"] = tprobs
test["Status"] = pd.to_numeric(test["Status"])
test0 = test[(test["Status"] <= 45) & (test["Goal"] == 0)]
test0 = test0[["MatchID","HomeTeamName","GuestTeamName","Status","Goal","Prediction","Probability","HostonTargetFT","GuestonTargetFT","HostCornersFT","GuestCornersFT",
       "HostYCFT","GuestYCFT","HostRCFT",
      "GuestRCFT","HostAttacksFT","GuestAttacksFT","HostoffTargetFT",
      "GuestoffTargetFT","HostPossessionFT","GuestPossessionFT"]]

test0.sort_values("Probability", ascending = False)

testj = test0[["MatchID","Probability"]]
testj.to_json()

import flask
from flask import request

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/test')
def test():
    return testj

app.run()

# flask depends on this env variable to find the main file
#export FLASK_APP=flasktrial.py

# now we just need to ask flask to run
#flask run











