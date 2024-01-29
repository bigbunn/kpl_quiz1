import streamlit as st
import numpy as np
import pandas as pd
import os
import pickle
from datetime import datetime, date
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from forex_python.converter import CurrencyRates
import locale

def dollar_to_rupiah(dollar):
    c = CurrencyRates()
    rupiah = c.convert('USD','IDR',float(dollar))
    return rupiah

def load_model(filename,datasetpath):
    dataset=pd.read_csv(datasetpath)
    df=pd.DataFrame(dataset,columns=["YearsExperience","Salary"])

    x=df["YearsExperience"].values.reshape(-1,1)
    y=df["Salary"].values.reshape(-1,1)

    x_train,x_test,y_train,y_test=train_test_split(x,y)

    lr=LinearRegression()
    lr.fit(x_train,y_train)

    y_pred=lr.predict(x_test)
    print("Model training successfull")
    print("R2 Score: ",metrics.r2_score(y_test,y_pred))

    pickle.dump(lr,open(filename,"wb"))

def predict(model,year):
    pred = model.predict(year)
    return pred

def main(filename):
    
    st.title("Analisis Prediksi Gaji")
    st.write("Gaji anda akan dinilai berdasarkan berapa tahun pengalaman anda : ")
    loaded_model = pickle.load(open(filename,'rb'))

    date = st.date_input("Kapan anda pertama kali bekerja?")
    datenow= date.today()
    tahun = datenow.year - date.year

    if tahun < 0:
        st.write("Harap masukkan tahun yang sesuai")
    else:
        st.write("Anda sudah bekerja selama ",tahun," tahun")
        year = [[float(tahun)]]
        result = predict(loaded_model,year)
        
        gaji_dalam_dollar = (result[0,0])
        gaji_dalam_rupiah = (dollar_to_rupiah(gaji_dalam_dollar))

        locale.setlocale(locale.LC_ALL,'usa_USA')
        dollar_curency=locale.currency(gaji_dalam_dollar,grouping=True,symbol=True)

        locale.setlocale(locale.LC_ALL,'id_ID')
        rupiah_curency=locale.currency(gaji_dalam_rupiah,grouping=True,symbol=True)
        st.write("Perkiraan Gaji : ",dollar_curency," atau ",rupiah_curency)


if __name__ == "__main__":
    filename = "mymodel.sav"
    datasetpath="Salary_dataset.csv"

    if (os.path.exists(filename)):
        main(filename)
    else:
        if os.path.exists(datasetpath):
            load_model(filename,datasetpath)
            main(filename)

    
        
