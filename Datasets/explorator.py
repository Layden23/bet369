# -*- coding: utf-8 -*-
"""explorator.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1UsHA1Fe7FinqSQlZmfkiEC8BrkDNyIMa
"""

import urllib.request, json, time, os 
import pandas as pd
from datetime import datetime

while 1:
    #Open webpage
    with urllib.request.urlopen("https://young-mountain-84413.herokuapp.com/goles") as url:
        live = json.loads(url.read().decode())
    
    nlive = []
    #Get matches that have more that have our info:
    variables = ("id" and "league" and "host" and "guest" and "rd" and "plus" and "sdh" and "sdf" and "status")
    for i in live:
        if variables in i:
            if "rd" in i:
                if "sd" in i:
                    if "plus" in i:
                        if "events" in i:
                            nlive.append(i)
                
    df = pd.DataFrame()
    df2 = pd.DataFrame()
    df3 = pd.DataFrame()
    n = range(len(nlive))
    for i in n:
        a = pd.DataFrame({nlive[i]["id"]},index= [i])
        a = a.rename(columns = {0:"MatchID"})
        league = pd.DataFrame(nlive[i]["league"], index = [i])
        league =  league.rename(columns = {'i': 'LeagueID', 'zc': 'ZC', 'jc': 'JC', 'bd': 'BD', 'n':'LeagueName',
          'fn': 'LeagueFullname', 'ls': 'LeagueAbbreviation', 'sbn': 'SBN', 'stn': 'STN', 'ci': 'CountryID',
          'cn': 'CountryName', 'cs': 'CountryAbbreviation'})
        host = pd.DataFrame(nlive[i]["host"], index = [i])
        host = host.rename(columns={'i': 'HomeTeamID', 'n':'HomeTeamName',
          'sbn': 'SBN', 'stn': 'STN'})
        guest = pd.DataFrame(nlive[i]["guest"], index = [i])
        guest = guest.rename(columns = {'i': 'GuestTeamID', 'n': 'GuestTeamName', 'sbn': 'SBN', 'stn': 'STN'})
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
        if (nlive[i]["rd"]["gg"]== "0" and nlive[i]["rd"]["hg"]== "0"):
            label = pd.DataFrame({"Goal": 0}, index = [i])
        else:
            label = pd.DataFrame({"Goal": 1}, index = [i])
    
    
        df = pd.concat([a,league,host,guest,rd,plus,sdf,sdh, status, label], axis = 1)
        df3 = df.loc[:,['MatchID', 'LeagueID', 'LeagueName', 'LeagueFullname', 'LeagueAbbreviation', 'CountryID',
                   'CountryName', 'CountryAbbreviation', 'HomeTeamID', 'HomeTeamName', 'GuestTeamID', 'GuestTeamName',
                   'HostHandicapFT', 'ExpectedGoalsFT','BettingCuoteFT', 'HostHandicapHT','ExpectedGoalsHT',
                   'BettingCuoteHT', 'HostGoalsFT','GuestGoalsFT','HostCornersFT','GuestCornersFT','HostYCFT',
                   'GuestYCFT','HostRCFT', 'GuestRCFT', 'HostAttacksFT', 'GuestAttacksFT', 'HostDAttacksFT',
                   'GuestDAttacksFT', 'HostonTargetFT', 'GuestonTargetFT', 'HostoffTargetFT', 'GuestoffTargetFT',
                   'HostPossessionFT', 'GuestPossessionFT',"Status","Goal"]]
        df3["Pitch"] = nlive[i]["events"][-2]["c"]
        df3["Weather"] = nlive[i]["events"][-1]["c"]
        df2 = pd.concat([df2,df3])
        
    os.chdir('/Users/jojoel/Google Drive/Github/bet369/Datasets/')
    dirname = ('/Users/jojoel/Google Drive/Github/bet369/Datasets/'+datetime.now().strftime('%Y-%m-%d'))
    if not os.path.exists(dirname):
        os.mkdir(datetime.now().strftime('%Y-%m-%d'))
    dirname = ('/Users/jojoel/Google Drive/Github/bet369/Datasets/'+datetime.now().strftime('%Y-%m-%d')+'/')
    filename = datetime.now().strftime('gambling-%Y-%m-%d-%H-%M.csv')
    df2.to_csv(dirname+filename, index = False)
    time.sleep(30)

    