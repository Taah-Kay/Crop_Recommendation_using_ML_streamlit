import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import warnings

def load_model(modelfile):
	loaded_model = pickle.load(open(modelfile, 'rb'))
	return loaded_model

def main():
	#title
	html_temp = """
	<div>
	<h1 style="color:MEDIUMSEAGREEN;text-align:left;"> Crop Recommendation  🌱 </h1>
	</div>
	"""

	st.markdown(html_temp, unsafe_allow_html=True)

	col1, col2 = st.columns([2,2])

	with col1:
		with st.expander("Instruction", expanded=True):
			st.write("""
			Crop recommendation is a tool that helps farmers make informed decisions about the crops they should grow.
			It includes various factors like climate, soil type, humidity, Ph level and irrigation to provide personalized recommendation to farmers.
			""")

		'''
		## How does it work ❓
		Enter all the parameters and the machine learning model will predict the most suitable crops to grow in a particular farm based on various parameters
		'''

	with col2:
		st.subheader(" Find out the most suitable crop to grow in your farm 👨‍🌾")
		N = st.number_input("Nitrogen", 0,10000)
		P = st.number_input("Phosporus", 5,10000)
		K = st.number_input("Potassium", 5,10000)
		temp = st.number_input("Temperature", 0.0,10000.0)
		humidity = st.number_input("Humidity %", 0.0,10000.0)
		ph = st.number_input("PH", 0.0,10000.0)
		rainfall = st.number_input("Rainfall in mm", 0.0,10000.0)

		feature_list = [N, P, K, temp, humidity, ph, rainfall]
		single_pred = np.array(feature_list).reshape(1,-1)

		if st.button('Predict'):

			loaded_model = load_model('crmodel.pkl')
			prediction = loaded_model.predict(single_pred)
			col1.write('''
			## Results 🔍
			''')
			col1.success(f"{prediction.item().title()} are recommended for your farm.")

if __name__ == '__main__':
	main()