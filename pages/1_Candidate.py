import streamlit as st

st.set_page_config(page_title="Candidate Interview", layout="wide", initial_sidebar_state="collapsed")

st.title("🎤 Candidate Interview")

job_desc = st.text_area("Paste job requirements here:")

if st.button("Start Interview"):
    st.write("Here the questions will be displayed...")
    # Add your QCM/written question logic here
