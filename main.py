from pathlib import Path

code = r'''import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
st.set_page_config(
    page_title="펭귄 부리 분석",
    page_icon="🐧",
    layout="wide"
)

st.title("🐧 펭귄 부리로 종 추정하기")
st.write("펭귄의 부리 길이, 부리 깊이, 서식 섬 정보를 이용해 펭귄의 종을 분석하고 예측합니다.")

DATA_URL = (
    "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/"
    "main/inst/extdata/penguins.csv"
)

SPECIES_KR = {
    "Adelie": "아델리펭귄",
    "Chinstrap": "턱끈펭귄",
    "Gentoo": "젠투펭귄"
}

# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv(DATA_URL, encoding="utf-8")
    return data

try:
    original_df = load_data()
except Exception as e:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결이나 데이터 주소를 확인해 주세요.")
    st.exception(e)
    st.stop()

required_columns = ["species", "island", "bill_length_mm", "bill_depth_mm"]
missing_columns = [col for col in required_columns if col not in original_df.columns]

if missing_columns:
    st.error(f"데이터에 필요한 열이 없습니다: {missing_columns}")
    st.stop()

# 모델에 필요한 열의 결측치가 있는 행만 제거
df = original_df.dropna(subset=required_columns).copy()
df["species_kr"] = df["species"].map(SPECIES_KR)

# --------------------------------------------------
# 데이터 개요
# --------------------------------------------------
st.header("1. 데이터 살펴보기")

c1, c2, c3 = st.columns(3)
c1.metric("원본 데이터 행 수", f"{len(original_df)}개")
c2.metric("분석에 사용한 행 수", f"{len(df)}개")
c3.metric("제거된 행 수", f"{len(original_df) - len(df)}개")

with st.expander("데이터 앞부분 보기"):
    st.dataframe(df.head(10), use_container_width=True)

with st.expander("열별 결측치 확인"):
    missing_table = original_df.isna().sum().rename("결측치 개수").to_frame()
    st.dataframe(missing_table, use_container_width=True)
    st.caption("종, 섬, 부리 길이, 부리 깊이에 결측치가 있는 행은 모델 분석에서 제외했습니다. 성별 결측치는 모델 입력에 사용하지 않으므로 유지했습니다.")

with st.expander("데이터 형식 확인"):
    st.dataframe(
        original_df.dtypes.astype(str).rename("자료형").to_frame(),
        use_container_width=True
    )

# --------------------------------------------------
# 그래프 분석
# --------------------------------------------------
st.header("2. 펭귄 부리와 서식지 분석")

tab1, tab2, tab3, tab4 = st.tabs(
    ["종별 개체 수", "부리 길이와 깊이", "섬과 펭귄 종", "부리 길이 분포"]
)

with tab1:
    species_count = df["species_kr"].value_counts().rename_axis("펭귄 종").reset_index(name="개체 수")
    fig = px.bar(
        species_count,
        x="펭귄 종",
        y="개체 수",
        color="펭귄 종",
        title="펭귄 종별 개체 수",
        text="개체 수"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig = px.scatter(
        df,
        x="bill_length_mm",
        y="bill_depth_mm",
        color="species_kr",
        hover_data=["island"],
        labels={
            "bill_length_mm": "부리 길이(mm)",
            "bill_depth_mm": "부리 깊이(mm)",
            "species_kr": "펭귄 종"
        },
        title="부리 길이와 부리 깊이의 관계"
    )
    st.plotly_chart(fig, use_container_width=True)
    overall_corr = df["bill_length_mm"].corr(df["bill_depth_mm"])
    st.write(f"전체 펭귄의 부리 길이와 깊이 상관계수는 **{overall_corr:.3f}**입니다.")

    corr_by_species = (
        df.groupby("species_kr")[["bill_length_mm", "bill_depth_mm"]]
        .corr()
        .iloc[0::2, -1]
        .reset_index()
    )
    corr_by_species.columns = ["펭귄 종", "상관계수"]
    st.dataframe(corr_by_species, use_container_width=True)

with tab3:
    island_table = pd.crosstab(df["island"], df["species_kr"])
    st.write("섬에 따라 관찰된 펭귄 종의 분포입니다.")
    st.dataframe(island_table, use_container_width=True)
    island_plot = island_table.reset_index().melt(
        id_vars="island", var_name="펭귄 종", value_name="개체 수"
    )
    fig = px.bar(
        island_plot,
        x="island",
        y="개체 수",
        color="펭귄 종",
        barmode="group",
        labels={"island": "섬", "개체 수": "개체 수"},
        title="섬별 펭귄 종 분포"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("일부 종은 특정 섬에서만 관찰됩니다. 따라서 섬 정보가 예측에 큰 영향을 줄 수 있습니다.")

with tab4:
    fig = px.box(
        df,
        x="species_kr",
        y="bill_length_mm",
        color="species_kr",
        labels={
            "species_kr": "펭귄 종",
            "bill_length_mm": "부리 길이(mm)"
        },
        title="종별 부리 길이 분포"
    )
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# 모델 준비 및 비교
# --------------------------------------------------
st.header("3. 머신러닝 모델 비교")
st.write("세 모델을 같은 학습 데이터와 테스트 데이터로 평가합니다. 섬 정보는 원-핫 인코딩하여 모델에 입력합니다.")

X = df[["bill_length_mm", "bill_depth_mm", "island"]]
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

numeric_features = ["bill_length_mm", "bill_depth_mm"]
categorical_features = ["island"]

def make_preprocessor(scale_numeric=False):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(steps=numeric_steps)
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer(transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
    ])

models = {
    "로지스틱 회귀": Pipeline(steps=[
        ("preprocessor", make_preprocessor(scale_numeric=True)),
        ("model", LogisticRegression(max_iter=2000, random_state=42))
    ]),
    "결정 트리": Pipeline(steps=[
        ("preprocessor", make_preprocessor()),
        ("model", DecisionTreeClassifier(random_state=42))
    ]),
    "랜덤 포레스트": Pipeline(steps=[
        ("preprocessor", make_preprocessor()),
        ("model", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1
        ))
    ])
}

results = []
trained_models = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    trained_models[model_name] = model

    results.append({
        "모델": model_name,
        "정확도": accuracy_score(y_test, predictions),
        "정밀도(가중 평균)": precision_score(y_test, predictions, average="weighted", zero_division=0),
        "재현율(가중 평균)": recall_score(y_test, predictions, average="weighted", zero_division=0),
        "F1 점수(가중 평균)": f1_score(y_test, predictions, average="weighted", zero_division=0)
    })

results_df = pd.DataFrame(results).set_index("모델")
st.subheader("모델별 평가 결과")
st.dataframe(
    results_df.style.format("{:.3f}"),
    use_container_width=True
)
st.caption("표의 수치는 실제 데이터와 현재 코드의 학습·테스트 분할을 이용해 실행할 때 계산됩니다. 미리 정해 놓은 예시 수치가 아닙니다.")

metric_choice = st.selectbox(
    "비교할 평가 지표 선택",
    ["정확도", "정밀도(가중 평균)", "재현율(가중 평균)", "F1 점수(가중 평균)"]
)
metric_chart_df = results_df[[metric_choice]].reset_index()
fig = px.bar(
    metric_chart_df,
    x="모델",
    y=metric_choice,
    color="모델",
    text=metric_chart_df[metric_choice].map(lambda x: f"{x:.3f}"),
    title=f"모델별 {metric_choice} 비교",
    range_y=[0, 1]
)
st.plotly_chart(fig, use_container_width=True)

best_model_name = results_df["F1 점수(가중 평균)"].idxmax()
st.info(f"현재 테스트 데이터에서 F1 점수가 가장 높은 모델은 **{best_model_name}**입니다.")

with st.expander("평가 지표 뜻 알아보기"):
    st.markdown("""
    - **정확도(Accuracy):** 전체 예측 중 정답을 맞힌 비율입니다.
    - **정밀도(Precision):** 특정 종이라고 예측한 것 중 실제로 그 종인 비율입니다.
    - **재현율(Recall):** 실제 특정 종인 개체 중 모델이 찾아낸 비율입니다.
    - **F1 점수(F1-score):** 정밀도와 재현율을 함께 고려한 점수입니다.
    - **가중 평균(Weighted):** 각 종의 데이터 개수를 반영해 전체 점수를 계산하는 방식입니다.
    """)

# --------------------------------------------------
# 랜덤 포레스트 설명 및 중요도
# --------------------------------------------------
st.header("4. 랜덤 포레스트 분석")

rf_model = trained_models["랜덤 포레스트"]
rf_classifier = rf_model.named_steps["model"]
feature_names = rf_model.named_steps["preprocessor"].get_feature_names_out()
importance_df = pd.DataFrame({
    "입력 변수": feature_names,
    "중요도": rf_classifier.feature_importances_
}).sort_values("중요도", ascending=False)

fig = px.bar(
    importance_df,
    x="중요도",
    y="입력 변수",
    orientation="h",
    title="랜덤 포레스트 입력 변수 중요도"
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
**현재 랜덤 포레스트 주요 하이퍼파라미터**
- `n_estimators=300`: 학습에 사용할 결정 트리의 개수입니다.
- `max_depth=None`: 트리의 최대 깊이를 별도로 제한하지 않습니다.
- `min_samples_split=2`: 노드를 나누기 위해 필요한 최소 데이터 수입니다.
- `min_samples_leaf=1`: 마지막 노드에 있어야 하는 최소 데이터 수입니다.
- `random_state=42`: 실행할 때 결과를 재현하기 위한 기준값입니다.
""")

# --------------------------------------------------
# 새 펭귄 종 예측
# --------------------------------------------------
st.header("5. 새로운 펭귄 종 예측하기")
st.write("부리 길이, 부리 깊이, 섬을 선택하면 랜덤 포레스트 모델이 예상 종을 알려줍니다.")

col1, col2, col3 = st.columns(3)

with col1:
    bill_length = st.number_input(
        "부리 길이(mm)",
        min_value=30.0,
        max_value=60.0,
        value=45.0,
        step=0.1
    )

with col2:
    bill_depth = st.number_input(
        "부리 깊이(mm)",
        min_value=13.0,
        max_value=22.0,
        value=17.0,
        step=0.1
    )

with col3:
    island = st.selectbox(
        "관찰된 섬",
        sorted(df["island"].dropna().unique().tolist())
    )

if st.button("🐧 펭귄 종 예측하기", type="primary"):
    new_penguin = pd.DataFrame({
        "bill_length_mm": [bill_length],
        "bill_depth_mm": [bill_depth],
        "island": [island]
    })

    predicted_species = rf_model.predict(new_penguin)[0]
    predicted_name = SPECIES_KR.get(predicted_species, predicted_species)

    st.success(f"예측된 펭귄 종: {predicted_name}")
    st.write(f"입력한 부리 길이: **{bill_length:.1f} mm**")
    st.write(f"입력한 부리 깊이: **{bill_depth:.1f} mm**")
    st.write(f"선택한 섬: **{island}**")

    probabilities = rf_model.predict_proba(new_penguin)[0]
    probability_df = pd.DataFrame({
        "펭귄 종": [SPECIES_KR.get(name, name) for name in rf_classifier.classes_],
        "예측 확률": probabilities
    }).sort_values("예측 확률", ascending=False)

    fig = px.bar(
        probability_df,
        x="펭귄 종",
        y="예측 확률",
        color="펭귄 종",
        text=probability_df["예측 확률"].map(lambda x: f"{x:.1%}"),
        title="종별 예측 확률",
        range_y=[0, 1]
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("예측 확률은 학습된 모델의 추정값이며, 실제 종을 확정하는 값은 아닙니다.")

# --------------------------------------------------
# 마무리
# --------------------------------------------------
st.divider()
st.caption("데이터 출처: Palmer Penguins | 분석 목적: 부리 형태와 서식지 정보를 이용한 펭귄 종 분류")
'''

path = Path("/mnt/data/main.py")
path.write_text(code, encoding="utf-8")
print(f"생성 완료: {path} ({len(code.splitlines())}줄)")
