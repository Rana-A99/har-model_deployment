
import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

model = load_model('har_model.h5')

# List of activity labels 
activities = ['Stand', 'Sit', 'Talk-sit', 'Talk-stand','Stand-sit','Lay','Lay-stand','Pick','Jump','Push-up','Sit-up','Walk','Walk-backward','Walk-circle','Run','Stair-up','Stair-down','Table-tennis']

st.title("Human Activity Recognition")
st.write("Upload sensor data CSV file (300 timesteps x 6 features)")

uploaded_file = st.file_uploader("Choose CSV file", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file, delimiter=',')
    data = data.reshape(1, 300, 6)  
    
    # prediction
    prediction = model.predict(data)
    class_idx = np.argmax(prediction)
    
    # results
    st.subheader("Results:")
    st.write(f"Predicted Activity: **{activities[class_idx]}**")
    st.write(f"Confidence: **{prediction[0][class_idx]*100:.1f}%**")
