
import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

model = load_model('har_model.h5')

# List of activity labels 
activities = ['Stand', 'Sit', 'Talk-sit', 'Talk-stand','Stand-sit','Lay','Lay-stand','Pick','Jump','Push-up','Sit-up','Walk','Walk-backward','Walk-circle','Run','Stair-up','Stair-down','Table-tennis']

st.title("Human Activity Recognition")
st.write("Upload sensor data CSV file (300 timesteps x 6 features)")

df = pd.read_csv(uploaded_file, header=None)
data = df.to_numpy()

if data.shape == (300, 6):
    data = data.reshape(1, 300, 6)
elif data.shape == (1, 1800):
    data = data.reshape(1, 300, 6)
else:
    st.error("❌ CSV must have either 300 rows x 6 columns OR 1 row x 1800 columns.")
    st.stop()
  
    
    # prediction
    prediction = model.predict(data)
    class_idx = np.argmax(prediction)
    
    # results
    st.subheader("Results:")
    st.write(f"Predicted Activity: **{activities[class_idx]}**")
    st.write(f"Confidence: **{prediction[0][class_idx]*100:.1f}%**")
