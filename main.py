import streamlit as st

st.set_page_config(
    page_title="Kairo",
    page_icon="</>",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.ui.dashboard import main as dashboard_main

if __name__ == "__main__":
    dashboard_main()

