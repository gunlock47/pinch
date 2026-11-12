
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --------------------------------------------------
# 페이지 설정
# --------------------------------------------------

st.set_page_config(
    page_title="펭귄 부리 분석",
    page_icon="🐧",
    layout="wide"
)

# --------------------------------------------------
# 디자인
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;800&family=Pretendard:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Pretendard', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #071426, #101b35, #17152e);
    color: white;
}

h1, h2, h3, p, label {
    color: white !important;
}

.hero {
    padding: 35px;
    border: 1px solid #334b70;
    border-radius: 22px;
    background: linear-gradient(120deg, #102847, #25234b);
    text-align: center;
    margin-bottom: 25px;
}

.hero-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 38px;
    font-weight: 800;
    color: #55e6ff;
}

.hero-sub {
    color: #d5e4ff;
    font-size: 16px;
}

.section {
    padding: 20px;
    border-radius: 16px;
    background: #14233d;
    border: 1px solid #34496b;
    margin-bottom: 15px;
}

.metric-card {
    background: #1b3152;
    border: 1px solid #45658e;
    border-radius: 15px;
    padding: 20px;
    text-align: center;
}

.metric-title {
    color: #9cb8dc;
    font-size: 14px;
}

.metric-value {
    color: #55e6ff;
    font-size: 27px;
    font-weight: 800;
}

div[data-baseweb="select"] * {
    color: black !important;
}

div[data-baseweb="input"] input {
    color: black !important;
}

div[data-baseweb="select"] {
    background-color: white !important;
}

div[data-testid="stNumberInput"] input {
    color: black !important;
}

.notice {
    background: #e9f4ff;
    color: black;
    padding: 12px;
    border-radius: 8px;
    font-size: 14px;
}

/* 새로운 펭귄 종 예측하기 영역 */
.prediction-section {
    background: #eaf2ff;
    border: 2px solid #55e6ff;
    border-radius: 16px;
    padding: 22px;
    margin-top: 15px;
    margin-bottom: 15px;
}

.prediction-section h2,
.prediction-section h3,
.prediction-section p,
.prediction-section label {
    color: black !important;
}

/* 예측 버튼 */
div.stButton > button {
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 2px solid #55e6ff !important;
    border-radius: 10px !important;
    font-weight: 800 !important;
}

div.stButton > button p,
div.stButton > button span,
div.stButton > button div {
    color: #000000 !important;
}

div.stButton > button:hover {
    background-color: #d9f5ff !important;
    color: #000000 !important;
    border: 2px solid #55e6ff !important;
}

div.stButton > button:focus {
    color: #000000 !important;
    border-color: #55e6ff !important;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <div class="hero-title">PENGUIN LAB</div>
    <h2>🐧 펭귄 부리로 종 추정하기</h2>
    <div class="hero-sub">
        펭귄의 부리 형태와 서식지 데이터를 활용한 AI 종 분류 시스템
    </div>
</div>
""", unsafe_allow_html=True)

url = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv"

@st.cache_data
def load_data():
    return pd.read_csv(url, encoding="utf-8")

try:
    df = load_data()
except Exception:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결을 확인해 주세요.")
    st.stop()

species_korean = {
    "Adelie": "아델리펭귄",
    "Chinstrap": "턱끈펭귄",
    "Gentoo": "젠투펭귄"
}

island_korean = {
    "Biscoe": "비스코섬",
    "Dream": "드림섬",
    "Torgersen": "토르거슨섬"
}

# --------------------------------------------------
# 데이터 기본 정보
# --------------------------------------------------

st.header("📊 DATA OVERVIEW")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f'<div class="metric-card"><div class="metric-title">전체 데이터</div><div class="metric-value">{len(df)}</div></div>',
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f'<div class="metric-card"><div class="metric-title">펭귄 종류</div><div class="metric-value">{df["species"].nunique()}</div></div>',
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f'<div class="metric-card"><div class="metric-title">서식 섬</div><div class="metric-value">{df["island"].nunique()}</div></div>',
        unsafe_allow_html=True
    )

st.subheader("데이터 미리보기")
st.dataframe(df.head(10), use_container_width=True)

with st.expander("데이터 자료형 확인"):
    st.dataframe(df.dtypes.astype(str).rename("자료형"), use_container_width=True)

with st.expander("열별 결측치 확인"):
    missing = df.isna().sum().reset_index()
    missing.columns = ["열 이름", "결측치 개수"]
    st.dataframe(missing, use_container_width=True)
    st.write("전체 결측치 개수:", int(df.isna().sum().sum()))

# --------------------------------------------------
# 데이터 전처리
# --------------------------------------------------

st.header("🧹 DATA PREPROCESSING")

original_count = len(df)

model_df = df.dropna(
    subset=[
        "species",
        "island",
        "bill_length_mm",
        "bill_depth_mm"
    ]
).copy()

removed_count = original_count - len(model_df)

col1, col2 = st.columns(2)

with col1:
    st.metric("원본 데이터 행 수", original_count)

with col2:
    st.metric("결측치 제거 후 행 수", len(model_df))

st.write(f"모델 분석에 필요한 값이 없는 데이터 {removed_count}행을 제거했습니다.")
st.write("성별(sex)은 모델 입력에 사용하지 않으므로 성별 결측치는 제거하지 않았습니다.")

# 이상치 확인

st.subheader("이상치 확인")

fig_box = px.box(
    model_df,
    y="bill_length_mm",
    color="species",
    title="펭귄 종별 부리 길이 이상치 확인",
    labels={
        "bill_length_mm": "부리 길이(mm)",
        "species": "펭귄 종"
    }
)

fig_box.update_layout(template="plotly_dark")
st.plotly_chart(fig_box, use_container_width=True)

st.info(
    "상자그림에서 범위를 벗어난 값이 보이더라도 실제 펭귄의 개체 차이일 수 있으므로 "
    "임의로 삭제하지 않고 분석에 활용했습니다."
)

# --------------------------------------------------
# 펭귄 부리 형태 분석
# --------------------------------------------------

st.header("🔬 BEAK ANALYSIS")

st.subheader("1. 부리 길이와 부리 깊이의 관계")

fig_scatter = px.scatter(
    model_df,
    x="bill_length_mm",
    y="bill_depth_mm",
    color="species",
    hover_data=["island"],
    title="펭귄 종별 부리 길이와 깊이의 관계",
    labels={
        "bill_length_mm": "부리 길이(mm)",
        "bill_depth_mm": "부리 깊이(mm)",
        "species": "펭귄 종",
        "island": "서식 섬"
    }
)

fig_scatter.update_layout(template="plotly_dark")
st.plotly_chart(fig_scatter, use_container_width=True)

correlation = model_df["bill_length_mm"].corr(
    model_df["bill_depth_mm"]
)

st.metric("전체 부리 길이와 깊이의 상관계수", round(correlation, 3))

st.write(
    "전체 펭귄을 대상으로 계산한 상관계수입니다. "
    "종별로 나누어 분석하면 서로 다른 경향이 나타날 수 있습니다."
)

st.subheader("2. 펭귄 종별 부리 형태 비교")

species_beak = model_df.groupby("species")[
    ["bill_length_mm", "bill_depth_mm"]
].mean().reset_index()

species_beak["species"] = species_beak["species"].map(species_korean)

fig_beak = px.bar(
    species_beak,
    x="species",
    y=["bill_length_mm", "bill_depth_mm"],
    barmode="group",
    title="펭귄 종별 평균 부리 길이와 깊이",
    labels={
        "species": "펭귄 종",
        "value": "평균 길이(mm)",
        "variable": "부리 측정 항목"
    }
)

fig_beak.update_layout(template="plotly_dark")
st.plotly_chart(fig_beak, use_container_width=True)

# --------------------------------------------------
# 서식지 분석
# --------------------------------------------------

st.header("🌍 HABITAT ANALYSIS")

st.subheader("3. 섬별 펭귄 종 분포")

island_data = pd.crosstab(
    model_df["island"],
    model_df["species"]
).reset_index()

island_data["island"] = island_data["island"].map(island_korean)

fig_island = px.bar(
    island_data,
    x="island",
    y=["Adelie", "Chinstrap", "Gentoo"],
    barmode="group",
    title="서식하는 섬에 따른 펭귄 종 분포",
    labels={
        "island": "섬 이름",
        "value": "펭귄 수",
        "variable": "펭귄 종"
    }
)

fig_island.update_layout(template="plotly_dark")
st.plotly_chart(fig_island, use_container_width=True)

st.subheader("4. 펭귄 부리 형태와 서식지의 관계")

fig_habitat = px.scatter(
    model_df,
    x="bill_length_mm",
    y="bill_depth_mm",
    color="island",
    symbol="species",
    hover_data=["species"],
    title="서식지에 따른 펭귄 부리 형태",
    labels={
        "bill_length_mm": "부리 길이(mm)",
        "bill_depth_mm": "부리 깊이(mm)",
        "island": "서식 섬",
        "species": "펭귄 종"
    }
)

fig_habitat.update_layout(template="plotly_dark")
st.plotly_chart(fig_habitat, use_container_width=True)

# --------------------------------------------------
# 머신러닝: 랜덤 포레스트
# --------------------------------------------------

st.header("🤖 AI PREDICTION")
st.subheader("랜덤 포레스트 펭귄 종 분류 모델")

features = [
    "bill_length_mm",
    "bill_depth_mm",
    "island"
]

X = model_df[features]
y = model_df["species"]

numeric_features = [
    "bill_length_mm",
    "bill_depth_mm"
]

categorical_features = ["island"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

random_forest = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1
            )
        )
    ]
)

random_forest.fit(X_train, y_train)

y_pred = random_forest.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

st.markdown(
    '<div class="section"><h3>🌲 RANDOM FOREST MODEL</h3>'
    '<p>여러 결정 트리의 예측 결과를 종합하여 펭귄의 종을 분류합니다.</p></div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("정확도", f"{accuracy * 100:.2f}%")

with col2:
    st.metric("정밀도", f"{precision * 100:.2f}%")

with col3:
    st.metric("재현율", f"{recall * 100:.2f}%")

with col4:
    st.metric("F1 점수", f"{f1 * 100:.2f}%")

# --------------------------------------------------
# 새로운 펭귄 종 예측
# --------------------------------------------------

st.markdown(
    '<div class="prediction-section">'
    '<h2>🐧 새로운 펭귄 종 예측하기</h2>'
    '<p>부리 길이와 깊이, 서식하는 섬을 입력하면 랜덤 포레스트 모델이 펭귄 종을 예측합니다.</p>'
    '</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    bill_length = st.number_input(
        "부리 길이(mm)",
        min_value=25.0,
        max_value=70.0,
        value=45.0,
        step=0.1
    )

with col2:
    bill_depth = st.number_input(
        "부리 깊이(mm)",
        min_value=10.0,
        max_value=25.0,
        value=17.0,
        step=0.1
    )

selected_island = st.selectbox(
    "서식하는 섬",
    ["Biscoe", "Dream", "Torgersen"],
    format_func=lambda x: island_korean[x]
)

if st.button("🔍 펭귄 종 예측하기", use_container_width=True):

    input_data = pd.DataFrame({
        "bill_length_mm": [bill_length],
        "bill_depth_mm": [bill_depth],
        "island": [selected_island]
    })

    prediction = random_forest.predict(input_data)[0]
    probabilities = random_forest.predict_proba(input_data)[0]

    predicted_korean = species_korean[prediction]

    st.success(f"예측된 펭귄 종: {predicted_korean}")

    st.write(f"입력한 부리 길이: {bill_length} mm")
    st.write(f"입력한 부리 깊이: {bill_depth} mm")
    st.write(f"선택한 섬: {island_korean[selected_island]}")

    st.subheader("종별 예측 확률")

    probability_df = pd.DataFrame({
        "펭귄 종": [
            species_korean[s]
            for s in random_forest.classes_
        ],
        "예측 확률": probabilities
    })

    fig_probability = px.bar(
        probability_df,
        x="펭귄 종",
        y="예측 확률",
        color="펭귄 종",
        title="랜덤 포레스트 종별 예측 확률",
        text=probability_df["예측 확률"].apply(
            lambda x: f"{x * 100:.1f}%"
        )
    )

    fig_probability.update_layout(
        template="plotly_dark",
        yaxis_tickformat=".0%"
    )

    st.plotly_chart(fig_probability, use_container_width=True)

    st.markdown(
        '<div class="notice">예측 확률은 학습된 모델의 추정값이며, 실제 종을 확정하는 값은 아닙니다.</div>',
        unsafe_allow_html=True
    )

# --------------------------------------------------
# 마무리
# --------------------------------------------------

st.divider()

st.markdown("""
<div class="hero">
    <div class="hero-title">PENGUIN LAB</div>
    <h3>DATA • BIOLOGY • MACHINE LEARNING</h3>
    <p class="hero-sub">
        펭귄의 부리 형태와 서식지 데이터를 활용한 생명과학 데이터 분석
    </p>
</div>
""", unsafe_allow_html=True)
