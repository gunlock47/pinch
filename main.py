
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# =========================================
# 1. PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="PENGUIN LAB | AI SYSTEM",
    page_icon="🐧",
    layout="wide"
)

# =========================================
# 2. DESIGN CSS
# =========================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;800&family=Pretendard:wght@400;500;600;700;800&display=swap');

.stApp {
    background:
    radial-gradient(ellipse at 10% 0%, #172C59 0%, transparent 40%),
    radial-gradient(ellipse at 90% 30%, #29204E 0%, transparent 35%),
    #080D1C;
    color: #F2F6FF;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: #101A30;
}

h1, h2, h3, p, label, span {
    font-family: 'Pretendard', sans-serif;
}

h1, h2, h3 {
    color: #F5F8FF;
}

.main .block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

.hero {
    background: linear-gradient(
        120deg,
        #101D3D 0%,
        #163C65 50%,
        #37235F 100%
    );
    border: 1px solid #416C9D;
    border-radius: 25px;
    padding: 48px 35px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 15px 55px #0008;
}

.hero-label {
    color: #5DE4F5;
    letter-spacing: 5px;
    font-size: 12px;
    font-weight: 800;
}

.hero h1 {
    font-family: 'Orbitron', sans-serif;
    font-size: clamp(30px, 5vw, 55px);
    letter-spacing: 2px;
