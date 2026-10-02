import streamlit as st
import os
import pandas as pd

st.title("🚨 AI CCTV Object Detection Dashboard")

st.subheader("📸 Detection Snapshot")

image_path = "screenshots/detection.jpg"

if os.path.exists(image_path):
    st.image(image_path, width=700)
else:
    st.write("No detection image found")


st.subheader("📝 Detection History")

file_path = "detection/detections.txt"

if os.path.exists(file_path):

    with open(file_path, "r") as file:
        data = file.readlines()

    df = pd.DataFrame(data, columns=["Detection"])

    st.dataframe(df)

    st.success(f"Total Detections: {len(data)}")

else:
    st.warning("Detection file not found")