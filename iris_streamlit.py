import pandas as pd
import streamlit as st

st.write('Developed by Adil')

st.write("""
# Iris Flower Classifier web app 
App Test Phase 0
	""")
st.write('***')

st.write('***')
st.sidebar.header('USER INPUT PARAMS')
def user_input_features():
	sepal_length=st.sidebar.slider('sepal_length',4.3,7.9,5.4)
	sepal_width=st.sidebar.slider('sepal_width',2.0,4.4,3.4)
	petal_length=st.sidebar.slider('petal_length',1.0,6.9,1.3)
	petal_width=st.sidebar.slider('petal_width',0.1,2.5,0.2)
	data={'sepal_length':sepal_length,
		  'sepal_width':sepal_width,
		  'petal_length':petal_length,
		  'petal_width':petal_width}
	features=pd.DataFrame(data,index=[0])
	return features

df=user_input_features()

st.subheader('User Input Params')

st.write(df)

st.write("""
 Class Labels""")
labels_df=pd.DataFrame({0:'setosa',1:'versicolor',2:'virginica'},index=[0])


st.write(labels_df)
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

iris=datasets.load_iris()
x=iris.data
y=iris.target
model=RandomForestClassifier()
model.fit(x,y)
pred=model.predict(df)
pred_proba=model.predict_proba(df)

st.write('Prediction')
st.write('***')
st.write(iris.target_names[pred])
st.write('***')
st.write('Prob Pred')

st.write(pred_proba)


