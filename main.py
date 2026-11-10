
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

# ==========================================
# 1. 기본 설정
# ==========================================

st.set_page_config(
    page_title="펭귄 종 분류 연구소",
    page_icon="🐧",
    layout="wide"
)

# 디자인 CSS
st.markdown("""
<style>
.stApp {
    background-color: #F4F7FB;
}

.main-title {
    background: linear-gradient(120deg, #173B5E, #397CA5);
    padding: 35px;
    border-radius: 20px;
    color: white;
    text-align: center;
    margin-bottom: 25px;
}

.main-title h1 {
    color: white;
    font-size: 38px;
}

.main-title p {
    color: #E1F0FA;
    font-size: 17px;
}

.info-card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #DFE8F0;
    text-align: center;
    margin-bottom: 10px;
}

.result-card {
    background: #E5F3FA;
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    border: 1px solid #C5DFEE;
}

.small-caption {
    color: #64748B;
    font-size: 14px;
}

h2, h3 {
    color: #173B5E;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. 데이터 불러오기
# ==========================================

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv"

    data = pd.read_csv(url)

    data = data.dropna(
        subset=[
            "species",
            "island",
            "bill_length_mm",
            "bill_depth_mm"
        ]
    )

    return data


try:
    df = load_data()
except Exception:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결을 확인해 주세요.")
    st.stop()


# ==========================================
# 3. 머신러닝 모델 만들기
# ==========================================

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

    preprocessor = ColumnTransformer(
        transformers=[
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
        ]
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
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


# ==========================================
# 4. 상단 메인 화면
# ==========================================

st.markdown("""
<div class="main-title">
    <h1>🐧 PENGUIN CLASSIFIER</h1>
    <p>부리의 크기와 서식지를 이용한 펭귄 종 분류 연구소</p>
    <p>PALMER PENGUINS DATA SCIENCE PROJECT</p>
</div>
""", unsafe_allow_html=True)


# ==========================================
# 5. 데이터 요약
# ==========================================

st.subheader("📊 데이터 한눈에 보기")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("분석 데이터", f"{len(df)}마리")

with c2:
    st.metric("펭귄 종류", f"{df['species'].nunique()}종")

with c3:
    st.metric("서식 섬", f"{df['island'].nunique()}곳")

with c4:
    st.metric("모델 정확도", f"{accuracy * 100:.1f}%")

st.caption(
    "부리 길이, 부리 깊이, 서식 섬 정보를 이용하여 "
    "펭귄의 종을 예측합니다."
)

st.divider()


# ==========================================
# 6. 펭귄 종 예측하기
# ==========================================

st.header("🔍 나만의 펭귄 종 예측하기")

st.write(
    "부리의 크기와 펭귄이 발견된 섬을 입력하면 "
    "머신러닝 모델이 펭귄의 종을 예측합니다."
)

species_korean = {
    "Adelie": "아델리펭귄",
    "Chinstrap": "턱끈펭귄",
    "Gentoo": "젠투펭귄"
}

with st.container(border=True):

    st.subheader("① 부리 정보 입력")

    col1, col2 = st.columns(2)

    with col1:
        bill_length = st.number_input(
            "부리 길이 (mm)",
            min_value=20.0,
            max_value=70.0,
            value=45.0,
            step=0.1
        )

    with col2:
        bill_depth = st.number_input(
            "부리 깊이 (mm)",
            min_value=10.0,
            max_value=30.0,
            value=17.0,
            step=0.1
        )

    st.subheader("② 서식 섬 선택")

    island = st.selectbox(
        "펭귄이 발견된 섬",
        sorted(df["island"].unique()),
        format_func=lambda x: {
            "Biscoe": "비스코섬",
            "Dream": "드림섬",
            "Torgersen": "토르거슨섬"
        }.get(x, x)
    )

    predict_button = st.button(
        "🐧 펭귄 종 예측하기",
        use_container_width=True,
        type="primary"
    )


if predict_button:

    input_data = pd.DataFrame({
        "bill_length_mm": [bill_length],
        "bill_depth_mm": [bill_depth],
        "island": [island]
    })

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    korean_name = species_korean[prediction]

    st.markdown("---")
    st.subheader("🎉 예측 결과")

    st.markdown(
        f"""
        <div class="result-card">
            <h2>예측된 펭귄은?</h2>
            <h1 style="font-size:38px; color:#205B83;">
                🐧 {korean_name}
            </h1>
            <p>AI가 입력된 부리 정보와 서식지를 분석했습니다.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("### 종별 예측 확률")

    probability_df = pd.DataFrame({
        "펭귄 종류": [
            species_korean[x] for x in model.classes_
        ],
        "예측 확률": probabilities * 100
    }).sort_values("예측 확률", ascending=False)

    fig = px.bar(
        probability_df,
        x="펭귄 종류",
        y="예측 확률",
        text=probability_df["예측 확률"].map(
            lambda x: f"{x:.1f}%"
        ),
        color="펭귄 종류",
        title="펭귄 종별 예측 확률",
        color_discrete_sequence=[
            "#397CA5", "#65A9B9", "#F0B75E"
        ]
    )

    fig.update_layout(
        yaxis_title="예측 확률 (%)",
        xaxis_title="",
        showlegend=False,
        yaxis_range=[0, 100]
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "예측 확률은 모델의 추정값이며, 실제 펭귄의 종을 "
        "확정하는 결과는 아닙니다."
    )


# ==========================================
# 7. 데이터 시각화
# ==========================================

st.divider()
st.header("📈 펭귄 데이터 분석")

tab1, tab2, tab3 = st.tabs([
    "부리 크기 분석",
    "섬별 펭귄 분포",
    "데이터 살펴보기"
])


# 부리 크기 분석
with tab1:

    st.subheader("종에 따른 부리 크기 비교")

    fig1 = px.scatter(
        df,
        x="bill_length_mm",
        y="bill_depth_mm",
        color="species",
        hover_data=["island"],
        labels={
            "bill_length_mm": "부리 길이 (mm)",
            "bill_depth_mm": "부리 깊이 (mm)",
            "species": "펭귄 종류"
        },
        color_discrete_sequence=[
            "#397CA5", "#E6A34C", "#54A89A"
        ],
        title="펭귄 종별 부리 길이와 깊이"
    )

    fig1.update_traces(marker=dict(size=9, opacity=0.8))

    st.plotly_chart(fig1, use_container_width=True)

    st.caption(
        "점 하나는 펭귄 한 마리를 의미합니다. "
        "색깔별로 펭귄 종을 구분할 수 있습니다."
    )


# 섬별 분포
with tab2:

    st.subheader("섬에 따른 펭귄 종 분포")

    island_count = pd.crosstab(
        df["island"],
        df["species"]
    ).reset_index()

    island_count = island_count.rename(
        columns={
            "Biscoe": "아델리펭귄·젠투펭귄",
            "Dream": "턱끈펭귄·아델리펭귄",
            "Torgersen": "아델리펭귄"
        }
    )

    # 실제 종별 분포를 정확히 표시
    island_count = pd.crosstab(
        df["island"],
        df["species"]
    ).reset_index()

    island_count = island_count.melt(
        id_vars="island",
        var_name="species",
        value_name="개체 수"
    )

    island_count["species"] = island_count["species"].map(
        species_korean
    )

    fig2 = px.bar(
        island_count,
        x="island",
        y="개체 수",
        color="species",
        barmode="group",
        title="서식 섬에 따른 펭귄 종류",
        labels={
            "island": "섬",
            "species": "펭귄 종류"
        },
        color_discrete_sequence=[
            "#397CA5", "#E6A34C", "#54A89A"
        ]
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.caption(
        "각 섬에 어떤 종류의 펭귄이 많이 분포하는지 비교할 수 있습니다."
    )


# 데이터 표
with tab3:

    st.subheader("원본 데이터 확인")

    st.dataframe(
        df[
            [
                "species",
                "island",
                "bill_length_mm",
                "bill_depth_mm"
            ]
        ].rename(
            columns={
                "species": "펭귄 종류",
                "island": "서식 섬",
                "bill_length_mm": "부리 길이(mm)",
                "bill_depth_mm": "부리 깊이(mm)"
            }
        ),
        use_container_width=True
    )


# ==========================================
# 8. 모델 성능 확인
# ==========================================

st.divider()
st.header("🧠 머신러닝 모델 성능")

st.write(
    "학습한 모델이 새로운 시험 데이터에서 "
    "얼마나 정확하게 펭귄 종을 분류했는지 확인합니다."
)

m1, m2 = st.columns(2)

with m1:
    st.metric(
        "시험 데이터 정확도",
        f"{accuracy * 100:.2f}%"
    )

with m2:
    st.metric(
        "시험 데이터 개수",
        f"{len(y_test)}마리"
    )

st.subheader("혼동 행렬")

cm_fig = px.imshow(
    matrix,
    x=[species_korean[x] for x in model.classes_],
    y=[species_korean[x] for x in model.classes_],
    text_auto=True,
    color_continuous_scale="Blues",
    labels={
        "x": "예측한 종",
        "y": "실제 종",
        "color": "개체 수"
    },
    title="실제 종과 예측한 종 비교"
)

st.plotly_chart(cm_fig, use_container_width=True)

st.caption(
    "대각선에 있는 숫자가 클수록 실제 종을 정확하게 "
    "맞힌 개체가 많다는 의미입니다."
)


# ==========================================
# 9. 마무리
# ==========================================

st.divider()

st.markdown("""
<div class="main-title">
    <h2>🐧 PENGUIN RESEARCH</h2>
    <p>부리와 서식지 정보를 통해 펭귄의 특징을 알아보았습니다.</p>
    <p>DATA SCIENCE PROJECT</p>
</div>
""", unsafe_allow_html=True)
