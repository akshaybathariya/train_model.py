import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

# Load model and dataset
model = joblib.load('iris_model.pkl')
iris = load_iris()

st.title("🌸 Iris Flower Predictor")
st.markdown("Enter flower features to predict the species.")

# Input sliders
sepal_length = st.slider('Sepal Length (cm)', 4.0, 8.0, 5.1)
sepal_width = st.slider('Sepal Width (cm)', 2.0, 4.5, 3.5)
petal_length = st.slider('Petal Length (cm)', 1.0, 7.0, 1.4)
petal_width = st.slider('Petal Width (cm)', 0.1, 2.5, 0.2)

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_data)[0]
predicted_species = iris.target_names[prediction]

st.success(f"🌼 Predicted Species: **{predicted_species}**")

st.subheader("Your Input Summary")
st.dataframe(pd.DataFrame(input_data, columns=iris.feature_names))

st.subheader("Class Distribution in Original Dataset")
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
st.bar_chart(df['species'].value_counts())
