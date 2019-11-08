import  pandas as pd
import pickle
import urllib.request, json
from flask import Flask,jsonify


app = Flask(__name__)
@app.route("/prob")
def hello():
    #Open webpage
    with urllib.request.urlopen("https://young-mountain-84413.herokuapp.com/goles") as url:
        live = json.loads(url.read().decode())

    nlive = []
    #Get matches that have more that have our info:
    for i in live:
        if "rd" in i:
            if "sd" in i:
                if "plus" in i:
                    nlive.append(i)

    df = pd.DataFrame()
    df2 = pd.DataFrame()
    test = pd.DataFrame()
    n = range(len(nlive))
    for i in n:
        a = pd.DataFrame({nlive[i]["id"]},index= [i])
        a = a.rename(columns = {0:"MatchID"})
        rd = pd.DataFrame(nlive[i]["rd"], index = [i])
        rd = rd.rename(columns={ 'hg': 'HostGoalsFT', 'gg': 'GuestGoalsFT',
            'hc': 'HostCornersFT', 'gc': 'GuestCornersFT', 'hy': 'HostYCFT', 'gy': 'GuestYCFT', 'hr': 'HostRCFT', 'gr': 'GuestRCFT'})
        plus = pd.DataFrame(nlive[i]["plus"], index = [i])
        plus = plus.rename(columns={ 'ha': 'HostAttacksFT', 'ga': 'GuestAttacksFT', 'hd': 'HostDAttacksFT',
            'gd': 'GuestDAttacksFT', 'hso': 'HostonTargetFT', 'gso': 'GuestonTargetFT', 'hsf': 'HostoffTargetFT',
            'gsf': 'GuestoffTargetFT', 'hqq': 'HostPossessionFT','gqq': 'GuestPossessionFT'})
        sdh = pd.DataFrame(nlive[i]["sd"]["h"], index = [i])
        sdh = sdh.rename(columns={'hrf': 'HostHandicapHT',
            'hdx': 'ExpectedGoalsHT', 'hcb': 'BettingCuoteHT'})
        sdf = pd.DataFrame(nlive[i]["sd"]["f"], index = [i])
        sdf = sdf.rename(columns={'hrf': 'HostHandicapFT', 'hdx': 'ExpectedGoalsFT', 'hcb': 'BettingCuoteFT'})
        status = pd.DataFrame({nlive[i]["status"]},index= [i])
        status = status.rename(columns = {0:"Status"})


        df = pd.concat([a,rd,plus,sdf,sdh, status], axis = 1)
        df2 = df.loc[:,['MatchID','HostCornersFT','GuestCornersFT','HostYCFT',
                       'GuestYCFT','HostRCFT', 'GuestRCFT', 'HostAttacksFT', 'GuestAttacksFT', 'HostDAttacksFT',
                       'GuestDAttacksFT', 'HostonTargetFT', 'GuestonTargetFT', 'HostoffTargetFT', 'GuestoffTargetFT',
                       'HostPossessionFT', 'GuestPossessionFT',"Status"]]
        test = pd.concat([test,df2])

    # load the model from disk
    rf = pickle.load(open('finalized_model.sav', 'rb'))
    test = test.dropna()
    # test = test.loc[(test['HostPossessionFT'] != 0) & (test['GuestPossessionFT'] != 0)]
    test = test.drop(test[test["Status"] == "HT"].index)
    test = test.drop(test[test["Status"] == "NS"].index)
    test = test.drop(test[test["Status"] == "FT"].index)
    ntest = test.loc[:,['HostCornersFT','GuestCornersFT','HostYCFT',
                       'GuestYCFT','HostRCFT', 'GuestRCFT', 'HostAttacksFT', 'GuestAttacksFT', 'HostDAttacksFT',
                       'GuestDAttacksFT', 'HostonTargetFT', 'GuestonTargetFT', 'HostoffTargetFT', 'GuestoffTargetFT',
                       'HostPossessionFT', 'GuestPossessionFT',"Status"]]
    for i in ntest.columns:
        ntest[i] = pd.to_numeric(ntest[i])
    test["Probability"] = rf.predict_proba(ntest)[:,1]
    test = test[["MatchID","Probability"]]
    testdict = test.to_dict('r')
    return jsonify(testdict)
if __name__ == "__main__":
    app.run()

