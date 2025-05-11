import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('har_model.h5')

activities = ['Stand', 'Sit', 'Talk-sit', 'Talk-stand','Stand-sit','Lay','Lay-stand','Pick','Jump','Push-up','Sit-up','Walk','Walk-backward','Walk-circle','Run','Stair-up','Stair-down','Table-tennis']

st.title("Human Activity Recognition")
st.write("Upload sensor data CSV file (300 timesteps x 6 features OR 1 row × 1800 values)")

uploaded_file = st.file_uploader("Choose CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, header=None)
        data = df.to_numpy()

        if data.shape == (300, 6):
            data = data.reshape(1, 300, 6)
        elif data.shape == (1, 1800):
            data = data.reshape(1, 300, 6)
        else:
            st.error(f"❌ CSV shape {data.shape} is not supported. Expected (300,6) or (1,1800).")
            st.stop()

        prediction = model.predict(data)
        class_idx = np.argmax(prediction)

        st.subheader("Results:")
        st.write(f"Predicted Activity: **{activities[class_idx]}**")
        st.write(f"Confidence: **{prediction[0][class_idx]*100:.1f}%**")
    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")
