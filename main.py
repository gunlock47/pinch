
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# =========================================
# 1. PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="PENGUIN LAB | AI SYSTEM",
    page_icon="🐧",
    layout="wide"
)

# =========================================
# 2. FUTURE DESIGN CSS
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
    color: white;
    margin: 15px 0;
}

.hero p {
    color: #C6DCF4;
    font-size: 16px;
}

.hero-tag {
    display: inline-block;
    background: #56D9E522;
    border: 1px solid #56D9E5;
    border-radius: 30px;
    padding: 8px 15px;
    color: #75F0F5;
    font-size: 12px;
    letter-spacing: 2px;
}

.section-label {
    color: #59DDEB;
    font-family: 'Orbitron', sans-serif;
    font-size: 12px;
    letter-spacing: 3px;
}

.metric-card {
    background: linear-gradient(145deg, #172642, #10192D);
    border: 1px solid #304D70;
    border-radius: 17px;
    padding: 22px;
    min-height: 120px;
    box-shadow: 0 8px 25px #0005;
}

.metric-title {
    color: #93A9C9;
    font-size: 13px;
}

.metric-number {
    font-family: 'Orbitron', sans-serif;
    font-size: 29px;
    color: #5DE4F5;
    font-weight: 700;
    margin-top: 12px;
}

.panel {
    background: #111C31;
    border: 1px solid #293F60;
    border-radius: 20px;
    padding: 25px;
}

.result-panel {
    background: linear-gradient(120deg, #143D5B, #35235D);
    border: 1px solid #5DE4F5;
    border-radius: 20px;
    padding: 30px;
    text-align: center;
}

.result-name {
    color: #74F1F4;
    font-size: 36px;
    font-weight: 800;
}

div.stButton > button {
    background: linear-gradient(100deg, #12BBD0, #7865ED);
    color: white;
    border: none;
    border-radius: 13px;
    min-height: 52px;
    font-weight: 800;
    font-size: 16px;
    box-shadow: 0 5px 20px #13BCD055;
    transition: 0.2s;
}

div.stButton > button:hover {
    border: 1px solid #9AF8FF;
    color: white;
    transform: translateY(-2px);
}

.stTabs [data-baseweb="tab"] {
    color: #B4C7E3;
    font-weight: 700;
}

.stTabs [aria-selected="true"] {
    color: #61E6F1 !important;
}

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    background-color: #172642;
    border-color: #456080;
    color: white;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #304D70;
    border-radius: 12px;
}

hr {
    border-color: #30435F;
}

[data-testid="stMetric"] {
    background: #172642;
    border: 1px solid #304D70;
    padding: 15px;
    border-radius: 15px;
}

[data-testid="stMetricValue"] {
    color: #60E4F0;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# =========================================
# 3. DATA
# =========================================

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv"
    data = pd.read_csv(url)

    return data.dropna(
        subset=[
            "species",
            "island",
            "bill_length_mm",
            "bill_depth_mm"
        ]
    )

try:
    df = load_data()
except Exception:
    st.error("데이터를 불러오지 못했습니다.")
    st.stop()


# =========================================
# 4. MACHINE LEARNING
# =========================================

@st.cache_resource
def train_model(data):

    X = data[
        ["bill_length_mm", "bill_depth_mm", "island"]
    ]

    y = data["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    preprocessing = ColumnTransformer([
        (
            "numeric",
            StandardScaler(),
            ["bill_length_mm", "bill_depth_mm"]
        ),
        (
            "island",
            OneHotEncoder(handle_unknown="ignore"),
            ["island"]
        )
    ])

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=model.classes_
    )

    return model, accuracy, matrix, y_test, predictions


model, accuracy, matrix, y_test, predictions = train_model(df)


# =========================================
# 5. COLOR SYSTEM
# =========================================

colors = {
    "Adelie": "#35D6E8",
    "Chinstrap": "#B58AFF",
    "Gentoo": "#FFAD66"
}

korean = {
    "Adelie": "아델리펭귄",
    "Chinstrap": "턱끈펭귄",
    "Gentoo": "젠투펭귄"
}

island_korean = {
    "Biscoe": "비스코섬",
    "Dream": "드림섬",
    "Torgersen": "토르거슨섬"
}

plot_theme = "plotly_dark"


# =========================================
# 6. HERO
# =========================================

st.markdown("""
<div class="hero">
    <div class="hero-label">ANTARCTIC INTELLIGENCE SYSTEM</div>
    <h1>PENGUIN LAB.</h1>
    <p>AI POWERED SPECIES CLASSIFICATION</p>
    <p>
        부리의 형태와 서식지 데이터를 분석하여
        펭귄의 종을 예측하는 인공지능 연구 프로젝트
    </p>
    <span class="hero-tag">● SYSTEM ONLINE</span>
    <span class="hero-tag">PALMER PENGUINS DATABASE</span>
</div>
""", unsafe_allow_html=True)

st.write("")


# =========================================
# 7. METRICS
# =========================================

st.markdown(
    '<div class="section-label">01 / RESEARCH OVERVIEW</div>',
    unsafe_allow_html=True
)

st.subheader("연구 데이터 현황")

c1, c2, c3, c4 = st.columns(4)

metrics = [
    ("TOTAL SPECIMENS", f"{len(df)}"),
    ("SPECIES TYPES", "03"),
    ("ISLAND LOCATIONS", "03"),
    ("MODEL ACCURACY", f"{accuracy*100:.1f}%")
]

for col, (title, number) in zip(
    [c1, c2, c3, c4], metrics
):
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">{title}</div>
                <div class="metric-number">{number}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.write("")
st.divider()


# =========================================
# 8. PREDICTION
# =========================================

st.markdown(
    '<div class="section-label">02 / AI CLASSIFICATION</div>',
    unsafe_allow_html=True
)

st.header("🐧 AI 펭귄 종 예측 시스템")

st.write(
    "부리의 길이와 깊이, 그리고 서식 섬을 입력하여 "
    "AI가 펭귄의 종을 예측합니다."
)

with st.container(border=True):

    st.markdown("### INPUT PARAMETERS")

    col1, col2 = st.columns(2)

    with col1:
        bill_length = st.number_input(
            "부리 길이 (mm)",
            20.0, 70.0, 45.0, 0.1
        )

    with col2:
        bill_depth = st.number_input(
            "부리 깊이 (mm)",
            10.0, 30.0, 17.0, 0.1
        )

    island = st.selectbox(
        "서식 섬",
        ["Biscoe", "Dream", "Torgersen"],
        format_func=lambda x: island_korean[x]
    )

    st.write("")

    run = st.button(
        "⚡ RUN AI PREDICTION",
        use_container_width=True
    )


if run:

    input_data = pd.DataFrame({
        "bill_length_mm": [bill_length],
        "bill_depth_mm": [bill_depth],
        "island": [island]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.write("")

    st.markdown(
        f"""
        <div class="result-panel">
            <div class="section-label">PREDICTION COMPLETE</div>
            <div class="result-name">
                🐧 {korean[prediction]}
            </div>
            <p style="color:#D0DFF7;">
                AI SPECIES IDENTIFICATION RESULT
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("종별 예측 확률")

    prob_df = pd.DataFrame({
        "종": [korean[x] for x in model.classes_],
        "확률": probability * 100,
        "색상": [colors[x] for x in model.classes_]
    })

    fig = go.Figure()

    for _, row in prob_df.iterrows():
        fig.add_trace(go.Bar(
            x=[row["종"]],
            y=[row["확률"]],
            marker_color=row["색상"],
            text=[f'{row["확률"]:.1f}%'],
            textposition="outside"
        ))

    fig.update_layout(
        template=plot_theme,
        height=350,
        yaxis=dict(
            title="확률 (%)",
            range=[0, 115]
        ),
        showlegend=False,
        margin=dict(t=30, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "예측 확률은 학습된 모델의 추정값이며, "
        "실제 종을 확정하는 값은 아닙니다."
    )


# =========================================
# 9. DATA VISUALIZATION
# =========================================

st.divider()

st.markdown(
    '<div class="section-label">03 / DATA VISUALIZATION</div>',
    unsafe_allow_html=True
)

st.header("데이터 시각화 연구실")

tab1, tab2, tab3 = st.tabs([
    "🧬 부리 형태",
    "🌎 섬별 분포",
    "📋 데이터베이스"
])


# BILL SCATTER
with tab1:

    st.subheader("부리 길이와 깊이의 관계")

    fig = px.scatter(
        df,
        x="bill_length_mm",
        y="bill_depth_mm",
        color="species",
        color_discrete_map=colors,
        hover_data=["island"],
        template=plot_theme,
        labels={
            "bill_length_mm": "부리 길이 (mm)",
            "bill_depth_mm": "부리 깊이 (mm)",
            "species": "펭귄 종"
        }
    )

    fig.update_traces(
        marker=dict(size=11, opacity=0.85)
    )

    fig.update_layout(
        height=520,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,27,48,0.8)"
    )

    st.plotly_chart(fig, use_container_width=True)


# ISLAND CHART
with tab2:

    st.subheader("섬에 따른 펭귄 분포")

    island_df = pd.crosstab(
        df["island"],
        df["species"]
    ).reset_index()

    island_df = island_df.melt(
        id_vars="island",
        var_name="species",
        value_name="개체 수"
    )

    island_df["species"] = island_df["species"].map(korean)
    island_df["island"] = island_df["island"].map(island_korean)

    fig = px.bar(
        island_df,
        x="island",
        y="개체 수",
        color="species",
        color_discrete_map={
            korean[k]: colors[k] for k in colors
        },
        barmode="group",
        template=plot_theme
    )

    fig.update_layout(
        height=480,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,27,48,0.8)"
    )

    st.plotly_chart(fig, use_container_width=True)


# DATABASE
with tab3:

    st.subheader("PALMER PENGUINS DATABASE")

    display_df = df[
        [
            "species",
            "island",
            "bill_length_mm",
            "bill_depth_mm"
        ]
    ].copy()

    display_df["species"] = display_df["species"].map(korean)
    display_df["island"] = display_df["island"].map(island_korean)

    display_df.columns = [
        "펭귄 종",
        "서식 섬",
        "부리 길이 (mm)",
        "부리 깊이 (mm)"
    ]

    st.dataframe(
        display_df,
        use_container_width=True,
        height=420
    )


# =========================================
# 10. MODEL PERFORMANCE
# =========================================

st.divider()

st.markdown(
    '<div class="section-label">04 / MODEL PERFORMANCE</div>',
    unsafe_allow_html=True
)

st.header("🧠 AI 모델 성능 분석")

a, b = st.columns(2)

with a:
    st.metric(
        "TEST ACCURACY",
        f"{accuracy*100:.2f}%"
    )

with b:
    st.metric(
        "TEST DATA",
        f"{len(y_test)} specimens"
    )

st.subheader("CONFUSION MATRIX")

fig = px.imshow(
    matrix,
    x=[korean[x] for x in model.classes_],
    y=[korean[x] for x in model.classes_],
    text_auto=True,
    color_continuous_scale=[
        "#111C31",
        "#176B87",
        "#35D6E8"
    ],
    template=plot_theme,
    labels={
        "x": "예측한 종",
        "y": "실제 종",
        "color": "개체 수"
    }
)

fig.update_layout(
    height=430,
    paper_bgcolor="rgba(0,0,0,0)"
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "혼동 행렬의 대각선은 실제 종을 정확히 예측한 개체 수를 의미합니다."
)


# =========================================
# 11. FOOTER
# =========================================

st.markdown("""
<div style="
    text-align:center;
    padding:40px 10px;
    border-top:1px solid #30435F;
    margin-top:30px;
">
    <div style="
        font-family:Orbitron;
        color:#5DE4F5;
        letter-spacing:4px;
        font-size:18px;
    ">
        PENGUIN LAB.
    </div>
    <p style="color:#8196B5; font-size:12px;">
        ANTARCTIC AI RESEARCH PROJECT · 2026
    </p>
    <p style="color:#8196B5; font-size:12px;">
        POWERED BY MACHINE LEARNING
    </p>
</div>
""", unsafe_allow_html=True)
