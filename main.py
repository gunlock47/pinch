
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

/* ===== ALL TEXT: WHITE ===== */

.stApp,
.stApp p,
.stApp span,
.stApp label,
.stApp small,
.stApp li,
.stApp div,
.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stMarkdownContainer"] li,
.stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stWidgetLabel"],
.stApp [data-testid="stMetricLabel"],
.stApp [data-testid="stMetricValue"],
.stApp [data-testid="stMetricDelta"],
.stApp [data-testid="stDataFrame"],
.stApp [data-testid="stDataFrame"] *,
.stApp [data-testid="stTable"] *,
.stApp [data-baseweb="tab"],
.stApp [data-baseweb="select"] *,
.stApp [data-baseweb="input"] *,
.stApp input,
.stApp textarea,
.stApp button,
.stApp [role="option"],
.stApp [role="listbox"],
.stApp [data-testid="stExpander"] *,
.stApp [data-testid="stAlert"] * {
    color: #FFFFFF !important;
}

/* Headings */

.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp h5,
.stApp h6 {
    color: #FFFFFF !important;
}

/* Hero and cards */

.hero-label,
.hero h1,
.hero p,
.hero-tag,
.section-label,
.metric-title,
.metric-number,
.result-name,
.result-panel p,
div[style*="text-align:center"] *,
div[style*="font-family:Orbitron"] {
    color: #FFFFFF !important;
}

/* Input placeholders */

.stApp input::placeholder,
.stApp textarea::placeholder {
    color: #FFFFFF !important;
    opacity: 0.85 !important;
}

/* Tabs */

.stTabs [data-baseweb="tab"],
.stTabs [data-baseweb="tab"] p,
.stTabs [aria-selected="true"] {
    color: #FFFFFF !important;
}

/* 입력값과 섬 선택 글자: 검정색 */

.stApp [data-testid="stNumberInput"] input,
.stApp [data-testid="stSelectbox"] [data-baseweb="select"] *,
.stApp [data-testid="stSelectbox"] [data-baseweb="select"] input {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}

/* 펼친 섬 선택 목록 글자: 검정색 */

div[data-baseweb="popover"] [role="listbox"],
div[data-baseweb="popover"] [role="option"],
div[data-baseweb="popover"] [role="option"] *,
ul[role="listbox"] li,
ul[role="listbox"] li * {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}

/* 예측 확률 안내문: 검정색 */

.probability-note,
.probability-note * {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}

.probability-note {
    background-color: #EAF2F8;
    padding: 12px 16px;
    border-radius: 10px;
    font-size: 14px;
}

/* Plotly chart text */

.js-plotly-plot .plotly text,
.js-plotly-plot .plotly .xtick text,
.js-plotly-plot .plotly .ytick text,
.js-plotly-plot .plotly .legend text,
.js-plotly-plot .plotly .gtitle,
.js-plotly-plot .plotly .annotation-text {
    fill: #FFFFFF !important;
    color: #FFFFFF !important;
}

/* INPUT PARAMETERS: BLACK TEXT */

.stApp [data-testid="stNumberInput"] input {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}

.stApp [data-testid="stNumberInput"] input::placeholder {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    opacity: 1 !important;
}

.stApp [data-testid="stSelectbox"] [data-baseweb="select"] *,
.stApp [data-testid="stSelectbox"] [data-baseweb="select"] input {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
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

    # 세 알고리즘이 같은 학습/테스트 데이터를 사용하도록 분리

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # 모든 모델에 같은 전처리 적용

    def make_preprocessing():
        return ColumnTransformer([
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

    # 비교할 세 가지 알고리즘

    model_definitions = {
        "로지스틱 회귀": LogisticRegression(max_iter=1000),

        "결정 트리": DecisionTreeClassifier(
            random_state=42
        ),

        "랜덤 포레스트": RandomForestClassifier(
            n_estimators=300,
            random_state=42
        )
    }

    models = {}
    results = {}

    for model_name, classifier in model_definitions.items():

        pipeline = Pipeline([
            ("preprocessing", make_preprocessing()),
            ("classifier", classifier)
        ])

        pipeline.fit(X_train, y_train)

        predicted = pipeline.predict(X_test)

        models[model_name] = pipeline

        results[model_name] = {
            "accuracy": accuracy_score(
                y_test, predicted
            ),

            "precision": precision_score(
                y_test,
                predicted,
                average="weighted",
                zero_division=0
            ),

            "recall": recall_score(
                y_test,
                predicted,
                average="weighted",
                zero_division=0
            ),

            "f1": f1_score(
                y_test,
                predicted,
                average="weighted",
                zero_division=0
            ),

            "matrix": confusion_matrix(
                y_test,
                predicted,
                labels=pipeline.classes_
            ),

            "predictions": predicted
        }

    return models, results, y_test


models, model_results, y_test = train_model(df)

# 기존 예측 기능은 로지스틱 회귀를 기본으로 유지

model = models["로지스틱 회귀"]

accuracy = model_results["로지스틱 회귀"]["accuracy"]

matrix = model_results["로지스틱 회귀"]["matrix"]

predictions = model_results["로지스틱 회귀"]["predictions"]


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


def apply_white_plotly_text(fig):
    """그래프 안의 글자를 흰색으로 통일합니다."""

    fig.update_layout(
        font=dict(color="#FFFFFF"),
        title_font=dict(color="#FFFFFF"),
        legend=dict(font=dict(color="#FFFFFF")),
        xaxis=dict(
            tickfont=dict(color="#FFFFFF"),
            title_font=dict(color="#FFFFFF")
        ),
        yaxis=dict(
            tickfont=dict(color="#FFFFFF"),
            title_font=dict(color="#FFFFFF")
        )
    )

    fig.update_xaxes(
        tickfont=dict(color="#FFFFFF"),
        title_font=dict(color="#FFFFFF"),
        color="#FFFFFF"
    )

    fig.update_yaxes(
        tickfont=dict(color="#FFFFFF"),
        title_font=dict(color="#FFFFFF"),
        color="#FFFFFF"
    )

    fig.update_traces(textfont_color="#FFFFFF")

    return fig


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
    ("LOGISTIC ACCURACY", f"{accuracy*100:.1f}%")
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

    apply_white_plotly_text(fig)

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        <div class="probability-note">
            예측 확률은 학습된 모델의 추정값이며,
            실제 종을 확정하는 값은 아닙니다.
        </div>
        """,
        unsafe_allow_html=True
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

    apply_white_plotly_text(fig)

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

    apply_white_plotly_text(fig)

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

st.header("🧠 AI 모델 성능 비교")

st.write(
    "같은 학습 데이터와 테스트 데이터를 사용하여 "
    "세 가지 알고리즘의 분류 성능을 비교합니다."
)

st.caption(
    "비교 모델: 로지스틱 회귀 · 결정 트리 · 랜덤 포레스트"
)


# 성능 비교용 데이터프레임

score_rows = []

for model_name, result in model_results.items():

    score_rows.append({
        "알고리즘": model_name,
        "정확도": result["accuracy"] * 100,
        "정밀도": result["precision"] * 100,
        "재현율": result["recall"] * 100,
        "F1 점수": result["f1"] * 100
    })

score_df = pd.DataFrame(score_rows)


# 가장 높은 정확도를 기록한 모델

best_row = score_df.loc[score_df["정확도"].idxmax()]

best_model_name = best_row["알고리즘"]

m1, m2, m3 = st.columns(3)

with m1:
    st.metric(
        "가장 높은 정확도",
        f'{best_row["정확도"]:.2f}%'
    )

with m2:
    st.metric(
        "최고 성능 모델",
        best_model_name
    )

with m3:
    st.metric(
        "테스트 데이터",
        f"{len(y_test)}개"
    )


# 알고리즘별 성능 비교 그래프

st.subheader("알고리즘별 성능 비교 그래프")

score_long = score_df.melt(
    id_vars="알고리즘",
    var_name="평가 항목",
    value_name="점수 (%)"
)

fig = px.bar(
    score_long,
    x="알고리즘",
    y="점수 (%)",
    color="평가 항목",
    barmode="group",
    text_auto=".1f",
    template=plot_theme,
    color_discrete_sequence=[
        "#35D6E8",
        "#B58AFF",
        "#FFAD66",
        "#72D6A0"
    ]
)

fig.update_layout(
    height=480,
    yaxis=dict(
        title="점수 (%)",
        range=[0, 105]
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,27,48,0.8)",
    legend_title_text="평가 항목"
)

apply_white_plotly_text(fig)

st.plotly_chart(fig, use_container_width=True)


# 알고리즘별 성능 수치

st.subheader("알고리즘별 성능 수치")

display_scores = score_df.copy()

for column in ["정확도", "정밀도", "재현율", "F1 점수"]:

    display_scores[column] = display_scores[column].map(
        lambda value: f"{value:.2f}%"
    )

st.dataframe(
    display_scores,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "정밀도·재현율·F1 점수는 세 종의 결과를 "
    "가중 평균(weighted average)으로 계산했습니다."
)


# 혼동 행렬 비교

st.subheader("CONFUSION MATRIX 비교")

selected_model_name = st.selectbox(
    "혼동 행렬을 확인할 알고리즘",
    list(models.keys())
)

selected_matrix = model_results[selected_model_name]["matrix"]

selected_classes = models[selected_model_name].classes_

fig = px.imshow(
    selected_matrix,
    x=[korean[x] for x in selected_classes],
    y=[korean[x] for x in selected_classes],
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

apply_white_plotly_text(fig)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "혼동 행렬의 대각선은 실제 종을 정확히 예측한 개체 수를 의미합니다."
)

st.info(
    "주의: 세 모델 모두 부리 길이, 부리 깊이뿐 아니라 서식 섬 정보도 "
    "입력으로 사용합니다. 따라서 이 결과만으로 부리 형태만의 "
    "분류 성능이라고 해석할 수는 없습니다."
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
