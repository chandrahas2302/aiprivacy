 
## linear_model.py

import pandas as pd
import numpy as np
import phe as paillier
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, r2_score
import json

class linmodel:
  def __init__(self):
    pass
  def getResults(self):
    df=pd.read_csv('dataset.csv')
    
    Y = np.array(df['PercentSalaryHike'])
    X = np.array(df.drop('PercentSalaryHike', axis=1))
    print("size of dataframe:", df.size)
    print("shape of dataframe:", df.shape)
    # print(X)
    # print(Y)
    X.reshape(-1, 1)
    Y.reshape(-1, 1)
    print(X.shape)
    print(Y.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    reg = LinearRegression().fit(X_train, Y_train)

    Y_pred=reg.predict(X_test)
    RMSE=pow(mean_squared_error(Y_pred, Y_test),0.5)
    R2=r2_score(Y_pred, Y_test)
    acc = reg.score(X_test, Y_test)
    # conf_mat = confusion_matrix(Y_test, Y_pred)
    # print(y_pred)
    print("accuracy:", acc*100)
    print("RMSE:", RMSE)
    print("r2_score:", R2)
    # print("confusion_matrix:", conf_mat)

    
    return reg, Y_pred, RMSE, R2

  def getCoef(self):
    return self.getResults()[0].coef_


################# Driver Code 0 #######################

def model_main():
	  cof=linmodel().getCoef()
	  print(cof)
if __name__=='__main__':
	model_main()


## cust.py


def storeKeys():
	public_key, private_key = paillier.generate_paillier_keypair()
	keys={}
	keys['public_key'] = {'n': public_key.n}
	keys['private_key'] = {'p': private_key.p,'q':private_key.q}
	with open('custkeys.json', 'w') as file: 
		json.dump(keys, file)

#storeKeys() #run only once

def getKeys():
	with open('custkeys.json', 'r') as file: 
		keys=json.load(file)
		pub_key=paillier.PaillierPublicKey(n=int(keys['public_key']['n']))
		priv_key=paillier.PaillierPrivateKey(pub_key,keys['private_key']['p'],keys['private_key']['q'])
		return pub_key, priv_key 

def serializeData(public_key, data):
	encrypted_data_list = [public_key.encrypt(x) for x in data]
	encrypted_data={}
	encrypted_data['public_key'] = {'n': public_key.n}
	encrypted_data['values'] = [(str(x.ciphertext()), x.exponent) for x in encrypted_data_list]
	serialized = json.dumps(encrypted_data)
	return serialized

def loadAnswer():
    with open('answer.json', 'r') as file: 
      ans=json.load(file)
    answer=json.loads(ans)
    return answer

################# Driver Code 1 #######################



## servercalc.py


def getData():
	with open('data.json', 'r') as file: 
		d=json.load(file)
	data=json.loads(d)
	return data

def computeData():
	data=getData()
	print("data:\n", data)
	mycoef=linmodel().getCoef()
	print("mycoef:\n", mycoef)
	pk=data['public_key']
	print("data['public_key']:", data['public_key'])
	pubkey= paillier.PaillierPublicKey(n=int(pk['n']))
	print("pubkey:", pubkey)
	enc_nums_rec = [paillier.EncryptedNumber(pubkey, int(x[0], int(x[1]))) for x in data['values']]
	print("enc_nums_rec:\n", enc_nums_rec)
	results=sum([mycoef[i]*enc_nums_rec[i] for i in range(len(mycoef))])
	return results, pubkey

def serializeData_server():
	results, pubkey = computeData()
	encrypted_data={}
	encrypted_data['pubkey'] = {'n': pubkey.n}
	encrypted_data['values'] = (str(results.ciphertext()), results.exponent)
	serialized = json.dumps(encrypted_data)
	return serialized

################# Driver Code 2 #######################




    
################# Driver Code 3 #######################



    
#################################################################################################################################



import numpy as np
import pickle
import pandas as pd
import streamlit as st 

from PIL import Image
def hike_predict(MonthlyIncome, JobLevel, TotalWorkingYears, PerformanceRating): 
    
    storeKeys()
    
    
    pub_key, priv_key = getKeys()

    data = [MonthlyIncome, JobLevel, TotalWorkingYears, PerformanceRating]
    ndata = ['MonthlyIncome','JobLevel','TotalWorkingYears','PerformanceRating']
    

    datafile=serializeData(pub_key, data)
    with open('data.json', 'w') as file: 
        json.dump(datafile, file)
    
    if st.button("Show encrypted data"):
        obj=json.loads(datafile)
        st.write("public key obtained:",str(obj['public_key']))
        list=obj['values']
        for i in range(len(list)):
            st.write("The encrypted form of",ndata[i],"is",list[i])
    
    datafile=serializeData_server()
    with open('answer.json', 'w') as file:
        json.dump(datafile, file)
    
    
    
    answer_file=loadAnswer()
    answer_key=paillier.PaillierPublicKey(n=int(answer_file['pubkey']['n']))
    answer = paillier.EncryptedNumber(answer_key, int(answer_file['values'][0]), int(answer_file['values'][1]))
    
    if (answer_key==pub_key):
        return priv_key.decrypt(answer),answer


def app():
#     st.title("Hike Prediction Serivce")
    html_temp = """
    <div style="background-color:Azure;padding:10px">
    <h2 style="color:black;text-align:center;">HIKE PREDICTION SERVICE</h2>
    </div>
    """
    # MonthlyIncome,JobLevel,TotalWorkingYears,PerformanceRating,PercentSalaryHike
    st.markdown(html_temp,unsafe_allow_html=True)
    id = st.text_input("ID")
    MonthlyIncome = st.number_input('MonthlyIncome', min_value=1000, max_value=10000, value=1000, step=1)
#     MonthlyIncome = int(MonthlyIncome)
    JobLevel = st.number_input('JobLevel', min_value=1, max_value=5, value=1, step=1)
#     JobLevel = int(JobLevel)
    TotalWorkingYears = st.number_input('TotalWorkingYears', min_value=0, max_value=50, value=0, step=1)
#     TotalWorkingYears = int(TotalWorkingYears)
    PerformanceRating = st.number_input('PerformanceRating', min_value=1, max_value=5, value=1, step=1)
#     PerformanceRating = int(PerformanceRating)
     #PercentSalaryHike = st.number_input('PercentSalaryHike', min_value=0, max_value=1, value=0, step=1)
#     PercentSalaryHike = int(PercentSalaryHike)
    result=""
    result,encresult=hike_predict(MonthlyIncome,JobLevel,TotalWorkingYears,PerformanceRating)
    if st.button("Show encrypted result"):
        st.write(encresult)
    if st.button("Predict"):
        st.success('The output is ${}(in decrypted form)'.format(result*5))

if __name__=='__app__':
    app()






