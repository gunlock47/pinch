
import streamlit as st
import pandas as pd
import plotly.express as px

# --------------------------------
# 1. 기본 설정
# --------------------------------

st.set_page_config(
    page_title="펭귄 부리 분석",
    page_icon="🐧",
    layout="wide"
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "allisonhorst/palmerpenguins/main/"
    "inst/extdata/penguins.csv"
)

SPECIES_NAMES = {
    "Adelie": "아델리펭귄",
    "Chinstrap": "턱끈펭귄",
    "Gentoo": "젠투펭귄"
}

st.title("🐧 펭귄 부리로 종 추정하기")
st.write(
    "부리 길이와 깊이를 입력하면 실제 펭귄 데이터와 비교하여 "
    "가장 비슷한 펭귄 종을 찾아줍니다."
)

# --------------------------------
# 2. 데이터 불러오기
# --------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")

try:
    raw_df = load_data()
except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.code(str(e))
    st.stop()

# --------------------------------
# 3. 원본 데이터 확인
# --------------------------------

st.header("1. 원본 데이터 살펴보기")

st.write(
    f"전체 데이터: {raw_df.shape[0]}행 / "
    f"{raw_df.shape[1]}열"
)

st.subheader("데이터 처음 10개")
st.dataframe(raw_df.head(10), use_container_width=True)

st.subheader("데이터 자료형")
st.dataframe(
    raw_df.dtypes.astype(str).rename("자료형"),
    use_container_width=True
)

# 열별 결측치 개수와 비율 계산
missing_summary = pd.DataFrame({
    "결측치 개수": raw_df.isna().sum(),
    "결측치 비율(%)": (
        raw_df.isna().mean() * 100
    ).round(1)
})

st.subheader("열별 결측치 현황")
st.dataframe(missing_summary, use_container_width=True)

st.caption(
    "결측치 개수와 비율은 전처리 전 원본 데이터 344개를 기준으로 계산합니다."
)

# --------------------------------
# 4. 데이터 전처리
# --------------------------------

before_rows = len(raw_df)

df = raw_df.dropna(
    subset=[
        "bill_length_mm",
        "bill_depth_mm",
        "species"
    ]
).copy()

after_rows = len(df)
removed_rows = before_rows - after_rows

df["species_korean"] = df["species"].map(SPECIES_NAMES)

# --------------------------------
# 5. 전처리 로그
# --------------------------------

log_messages = [
    f"원본 데이터: {before_rows}개 행",
    f"부리 길이 결측치: {missing_summary.loc['bill_length_mm', '결측치 개수']}개",
    f"부리 깊이 결측치: {missing_summary.loc['bill_depth_mm', '결측치 개수']}개",
    f"성별 결측치: {missing_summary.loc['sex', '결측치 개수']}개",
    f"전처리 결과: {before_rows} → {after_rows}",
    f"제거된 행: {removed_rows}개",
    "island 변수는 종 추정에서 제외"
]

st.sidebar.header("데이터 전처리 로그")

for message in log_messages:
    st.sidebar.write("• " + message)

st.sidebar.caption(
    "성별 결측 행은 분석에 필요하지 않아 유지합니다."
)

# --------------------------------
# 6. 데이터 전처리 결과
# --------------------------------

st.header("2. 데이터 전처리 결과")

col1, col2, col3 = st.columns(3)

col1.metric("전처리 전 데이터", f"{before_rows}개")
col2.metric("전처리 후 데이터", f"{after_rows}개")
col3.metric("제거된 데이터", f"{removed_rows}개")

st.subheader("펭귄 종별 데이터 개수")

species_counts = df["species_korean"].value_counts()

species_ratio = (
    df["species_korean"].value_counts(normalize=True) * 100
).round(1)

count_df = pd.DataFrame({
    "데이터 개수": species_counts,
    "비율(%)": species_ratio
})

st.dataframe(count_df, use_container_width=True)

fig_count = px.bar(
    count_df.reset_index(),
    x="species_korean",
    y="데이터 개수",
    color="species_korean",
    title="펭귄 종별 데이터 개수",
    labels={
        "species_korean": "펭귄 종",
        "데이터 개수": "개체 수"
    }
)

st.plotly_chart(fig_count, use_container_width=True)

st.info(
    "턱끈펭귄은 전체 데이터의 약 19.8%로 "
    "세 종 중 데이터 비율이 가장 낮습니다. "
    "종별 데이터 수의 차이를 고려하여 결과를 해석해야 합니다."
)

# --------------------------------
# 7. 종과 섬의 관계
# --------------------------------

st.header("3. 펭귄 종과 서식 섬의 관계")

island_table = pd.crosstab(
    df["species_korean"],
    df["island"]
)

st.write("종별 섬 분포")
st.dataframe(island_table, use_container_width=True)

st.warning(
    "턱끈펭귄은 모두 Dream 섬, 젠투펭귄은 모두 Biscoe 섬에 "
    "분포합니다. 따라서 island를 입력 변수에 포함하면 "
    "부리 특징을 분석하지 않고도 종을 쉽게 구분할 수 있습니다. "
    "이번 프로젝트에서는 island를 종 추정에 사용하지 않습니다."
)

# --------------------------------
# 8. 상관관계 분석
# --------------------------------

st.header("4. 부리 길이와 깊이의 상관관계")

overall_corr = df[
    ["bill_length_mm", "bill_depth_mm"]
].corr().iloc[0, 1]

st.subheader("① 전체 펭귄 산점도")

fig1 = px.scatter(
    df,
    x="bill_length_mm",
    y="bill_depth_mm",
    title="전체 펭귄의 부리 길이와 깊이",
    labels={
        "bill_length_mm": "부리 길이(mm)",
        "bill_depth_mm": "부리 깊이(mm)"
    }
)

st.plotly_chart(fig1, use_container_width=True)

st.metric("전체 상관계수", f"{overall_corr:.3f}")

st.subheader("② 종별 색상 산점도")

fig2 = px.scatter(
    df,
    x="bill_length_mm",
    y="bill_depth_mm",
    color="species_korean",
    title="펭귄 종별 부리 길이와 깊이",
    labels={
        "bill_length_mm": "부리 길이(mm)",
        "bill_depth_mm": "부리 깊이(mm)",
        "species_korean": "펭귄 종"
    }
)

st.plotly_chart(fig2, use_container_width=True)

st.subheader("③ 종별 상관계수")

species_corr = df.groupby("species_korean").apply(
    lambda group: group["bill_length_mm"].corr(
        group["bill_depth_mm"]
    )
)

for species, corr in species_corr.items():
    st.write(f"**{species}**: {corr:.3f}")

st.info(
    "전체 데이터와 종별 데이터에서 상관관계의 방향이 "
    "다르게 나타날 수 있습니다. 종마다 부리의 크기와 형태가 "
    "다르기 때문이며, 전체적인 관계만으로 각 종 내부의 "
    "관계를 판단하기 어렵습니다."
)

# --------------------------------
# 9. 펭귄 종 추정 기능
# --------------------------------

st.header("5. 부리 길이와 깊이로 펭귄 종 추정하기")

st.write(
    "부리 길이와 깊이를 입력하면 데이터에 저장된 펭귄 중 "
    "가장 가까운 특징을 가진 개체를 찾아줍니다."
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

if st.button("펭귄 종 추정하기", use_container_width=True):

    length_std = df["bill_length_mm"].std()
    depth_std = df["bill_depth_mm"].std()

    distances = (
        ((df["bill_length_mm"] - bill_length) / length_std) ** 2
        +
        ((df["bill_depth_mm"] - bill_depth) / depth_std) ** 2
    ) ** 0.5

    closest_index = distances.idxmin()
    closest = df.loc[closest_index]

    result_species = closest["species_korean"]

    st.success(
        f"가장 비슷한 종은 **{result_species}**입니다!"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("추정 종", result_species)

    col2.metric(
        "비교된 부리 길이",
        f"{closest['bill_length_mm']:.1f} mm"
    )

    col3.metric(
        "비교된 부리 깊이",
        f"{closest['bill_depth_mm']:.1f} mm"
    )

    st.write(f"입력한 부리 길이: {bill_length:.1f} mm")
    st.write(f"입력한 부리 깊이: {bill_depth:.1f} mm")

    fig3 = px.scatter(
        df,
        x="bill_length_mm",
        y="bill_depth_mm",
        color="species_korean",
        title="입력한 부리 특징과 실제 펭귄 데이터 비교",
        labels={
            "bill_length_mm": "부리 길이(mm)",
            "bill_depth_mm": "부리 깊이(mm)",
            "species_korean": "펭귄 종"
        }
    )

    fig3.add_scatter(
        x=[bill_length],
        y=[bill_depth],
        mode="markers",
        name="입력값",
        marker=dict(
            color="red",
            size=15,
            symbol="star"
        )
    )

    st.plotly_chart(fig3, use_container_width=True)

    st.caption(
        "이 결과는 부리 길이와 깊이가 가장 가까운 데이터 한 개를 "
        "찾은 것입니다. 학습된 머신러닝 모델의 예측이나 "
        "확정적인 종 판별 결과는 아닙니다."
    )

# --------------------------------
# 10. 원본 데이터 확인
# --------------------------------

st.header("6. 펭귄 데이터 확인")

with st.expander("전처리된 데이터 전체 보기"):
    st.dataframe(df, use_container_width=True)

# --------------------------------
# 11. 데이터 출처
# --------------------------------

st.header("7. 데이터 출처")

st.markdown(
    """
    - 데이터셋: Palmer Penguins
    - 분석 대상: 아델리펭귄, 턱끈펭귄, 젠투펭귄
    - 사용 변수: 부리 길이(bill_length_mm), 부리 깊이(bill_depth_mm)
    - 제외 변수: island, sex
    - 데이터 출처: [Palmer Penguins GitHub](https://github.com/allisonhorst/palmerpenguins)
    """
)

st.caption(
    "현재 사용 데이터는 핀치새가 아닌 펭귄 데이터입니다. "
    "핀치새 분석은 추후 확장 주제로 계획합니다."
)
