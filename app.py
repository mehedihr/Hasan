#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import necessary libraries
import flask
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import lightgbm
import dill


# In[2]:


#load models into memory only 
with open('model_up.pkl', 'rb') as f:
    model = dill.load(f)


# In[3]:


#load scaler into memory only
with open('scaler_up.pkl', 'rb') as f:
    scaler = dill.load(f)


# In[4]:


# Craete some dictionery to change the categorical value into int.
sub_grade_to_num={'A1':0,'A2':1,'A3':2,'A4':3,'A5':4,'B1':5,'B2':6,'B3':7,'B4':8,'B5':9,'C1':10,'C2':11,'C3':12,'C4':13,'C5':14,'D1':15,'D2':16,'D3':17,'D4':18,'D5':19,'E1':20,'E2':21,'E3':22,'E4':23,'E5':24,'F1':25,'F2':26,'F3':27,'F4':28,'F5':29,'G1':30,'G2':31,'G3':32,'G4':33,'G5':34}
home_ownership_to_num = {'OWN': 5,'MORTGAGE': 4,'RENT': 3,'ANY': 2,'OTHER': 1, 'NONE':0 }
verification_status_num ={'Verified':0,'Not Verified':1}
application_type_to_num={'Individual':0,'Joint':1}
debt_settlement_flag_to_num = {'No':0,'Yes':1}
pub_rec_bankruptcies_to_Num = {'No':0,'Yes':1}
#addr_state_to_num = {'AK':0,'AL':1,'AR':2,'AZ':3,'CA':4,'CO':5,'CT':6,'DC':7,'DE':8,'FL':9,'GA':10,'HI':11,'IA':12,'ID':13,'IL':14,'IN':15,'KS':16,'KY':17,'LA':18,'MA':19,'MD':20,'ME':21,'MI':22,'MN':23,'MO':24,'MS':25,'MT':26,'NC':27,'ND':28,'NE':29,'NH':30,'NJ':31,'NM':32,'NV':33,'NY':34,'OH':35,'OK':36,'OR':37,'PA':38,'RI':39,'SC':40,'SD':41,'TN':42,'TX':43,'UT':44,'VA':45,'VT':46,'WA':47,'WI':48,'WV':49,'WY':50}
emp_length_to_num = {'Less than 1 Year':0,'1 Year':1,'2 Years':2,'3 Years':3,'4 Years':4,'5 Years':5,'6 Years':6,'7 Years':7,'8 Years':8,'9 Years':9,'10+ Years':10,}


# In[5]:


#Define app for website.
app = flask.Flask(__name__, template_folder='templates')


# In[6]:


#Route the index page of website.
@app.route('/')
def main():
    return (flask.render_template('index.html'))


# In[7]:


#Route the prediction page and define the parameter..
@app.route("/prediction", methods=['GET', 'POST'])
def prediction():
    
    if flask.request.method == 'GET':
        return (flask.render_template('prediction.html'))
    
    if flask.request.method =='POST':
        
        #get input from the website page.

        #fico score as integer
        fico_range_high = int(flask.request.form['fico_range_high'])
        #loan amount as integer
        loan_amnt = float(flask.request.form['loan_amnt'])
        #term as integer: 36 or 60
        term = int(flask.request.form['term'])
        #debt to income as float
        dti = float(flask.request.form['dti'])
        #home ownership as string
        home_ownership = flask.request.form['home_ownership']
        #number or mortgage accounts as integer
        mort_acc = int(flask.request.form['mort_acc'])
        #annual income as float
        annual_inc = float(flask.request.form['annual_inc'])
        #verification status as 0, 1
        verification_status = flask.request.form['verification_status']
        #revolving utilization as float
        revol_util = float(flask.request.form['revol_util'])
        #Sug-Grade as string
        sub_grade = flask.request.form['sub_grade']
        #Employment Length as string
        emp_length = flask.request.form['emp_length']
        #Application Type as Individual and Joint
        application_type = flask.request.form['application_type']
        #Previous Bankruptcies as Yes and No
        pub_rec_bankruptcies = flask.request.form['pub_rec_bankruptcies']
        #Previous Debt Settlement flag as Yes and No
        debt_settlement_flag = flask.request.form['debt_settlement_flag']
        #time since first credit line in months
        er_credit_open_date = pd.to_datetime(flask.request.form['er_credit_open_date'])
        #Define issue date as today.
        issue_d = pd.to_datetime("today")
        #number of days of credit history from the earliest credit line open date till today 
        credit_hist = issue_d - er_credit_open_date
        
        #Save the input as dataset.
        temp = pd.DataFrame(index=[1])

        temp['fico_range_high']=fico_range_high
        temp['loan_amnt']=loan_amnt
        temp['term']=term
        temp['dti']=dti
        temp['home_ownership']=home_ownership_to_num[home_ownership]
        temp['mort_acc']=mort_acc
        temp['annual_inc'] = annual_inc
        temp['verification_status']= verification_status_num[verification_status]
        temp['revol_util']=revol_util
        temp['sub_grade']=sub_grade_to_num[sub_grade]
        temp['emp_length'] = emp_length_to_num[emp_length]
        temp['application_type']=application_type_to_num[application_type]
        temp['pub_rec_bankruptcies']=pub_rec_bankruptcies_to_Num[pub_rec_bankruptcies]
        temp['debt_settlement_flag']=debt_settlement_flag_to_num[debt_settlement_flag]
        temp['credit_hist'] = credit_hist.days


        #create output for result. 
        output_dict= dict()
        output_dict['Annual Income'] = annual_inc
        output_dict['FICO High Score'] = fico_range_high
        output_dict['Loan Amount']=loan_amnt
        output_dict['Term']=term

        #deep copy in data 
        data = temp.copy()
        data_scaled = scaler.transform(data)
        
        #make prediction
        pred = model.predict(data_scaled)
        
        #Result of the prediction
        if pred ==1:
            res = 'Loan Denied! Customer may default.'
        else:
            res = 'Loan Approved!'

        #render form again and add prediction
        return flask.render_template('prediction.html',original_input=output_dict,result=res)


# In[8]:


#run the app.
if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




