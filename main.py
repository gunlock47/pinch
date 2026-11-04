
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# ==========================================
# 1. 기본 설정
# ==========================================

st.set_page_config(
    page_title="펭귄 부리 연구소",
    page_icon="🐧",
    layout="wide"
)

st.title("🐧 펭귄 부리 연구소")
st.markdown(
    "펭귄의 부리 길이와 깊이, 종, 서식지 데이터를 분석하는 웹사이트입니다."
)

DATA_URL = (
    "https://raw.githubusercontent.com/allisonhorst/"
    "palmerpenguins/main/inst/extdata/penguins.csv"
)

SPECIES_NAMES = {
    "Adelie": "아델리펭귄",
    "Chinstrap": "턱끈펭귄",
    "Gentoo": "젠투펭귄"
}

ISLAND_NAMES = {
    "Biscoe": "비스코섬",
    "Dream": "드림섬",
    "Torgersen": "토르거슨섬"
}


# ==========================================
# 2. 데이터 불러오기
# ==========================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


try:
    raw_df = load_data()
except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.code(str(e))
    st.stop()


# ==========================================
# 3. 원본 데이터 살펴보기
# ==========================================

st.header("1. 데이터 살펴보기")

col1, col2 = st.columns(2)

with col1:
    st.metric("전체 데이터 행", raw_df.shape[0])

with col2:
    st.metric("전체 데이터 열", raw_df.shape[1])

st.subheader("원본 데이터 미리보기")
st.dataframe(raw_df.head(10), use_container_width=True)

st.subheader("열별 데이터 유형")
st.dataframe(
    raw_df.dtypes.astype(str).rename("데이터 유형"),
    use_container_width=True
)

st.subheader("열별 결측치 현황")

missing_summary = pd.DataFrame({
    "결측치 개수": raw_df.isna().sum(),
    "결측치 비율(%)": (raw_df.isna().mean() * 100).round(1)
})

st.dataframe(missing_summary, use_container_width=True)


# ==========================================
# 4. 데이터 전처리
# ==========================================

st.header("2. 데이터 전처리")

before_rows = len(raw_df)

df = raw_df.dropna(
    subset=["bill_length_mm", "bill_depth_mm", "species", "island"]
).copy()

after_rows = len(df)
removed_rows = before_rows - after_rows

df["species_korean"] = df["species"].map(SPECIES_NAMES)
df["island_korean"] = df["island"].map(ISLAND_NAMES)

df["bill_ratio"] = (
    df["bill_length_mm"] / df["bill_depth_mm"]
).round(2)

st.write(f"전처리 전 데이터: {before_rows}개")
st.write(f"전처리 후 데이터: {after_rows}개")
st.write(f"제거된 데이터: {removed_rows}개")

st.caption(
    "부리 길이, 부리 깊이, 종, 섬 정보가 없는 행을 제거했습니다. "
    "성별 등 분석에 사용하지 않는 열의 결측치는 유지했습니다."
)


# ==========================================
# 5. 펭귄 종 분포
# ==========================================

st.header("3. 펭귄 종 분포 분석")

species_counts = (
    df["species_korean"]
    .value_counts()
    .rename_axis("펭귄 종")
    .reset_index(name="개체 수")
)

fig_species = px.bar(
    species_counts,
    x="펭귄 종",
    y="개체 수",
    color="펭귄 종",
    text="개체 수",
    title="펭귄 종별 개체 수"
)

st.plotly_chart(fig_species, use_container_width=True)

st.dataframe(species_counts, use_container_width=True)


# ==========================================
# 6. 종별 부리 형태 분석
# ==========================================

st.header("4. 종별 부리 형태 분석")

col1, col2 = st.columns(2)

with col1:
    fig_length = px.box(
        df,
        x="species_korean",
        y="bill_length_mm",
        color="species_korean",
        points="all",
        title="펭귄 종별 부리 길이 비교",
        labels={
            "species_korean": "펭귄 종",
            "bill_length_mm": "부리 길이(mm)"
        }
    )
    st.plotly_chart(fig_length, use_container_width=True)

with col2:
    fig_depth = px.box(
        df,
        x="species_korean",
        y="bill_depth_mm",
        color="species_korean",
        points="all",
        title="펭귄 종별 부리 깊이 비교",
        labels={
            "species_korean": "펭귄 종",
            "bill_depth_mm": "부리 깊이(mm)"
        }
    )
    st.plotly_chart(fig_depth, use_container_width=True)


# ==========================================
# 7. 서식하는 섬과 펭귄 종의 관계
# ==========================================

st.header("5. 섬에 따른 펭귄 종 분포")

island_counts = (
    df.groupby(["island_korean", "species_korean"])
    .size()
    .reset_index(name="개체 수")
)

fig_island = px.bar(
    island_counts,
    x="island_korean",
    y="개체 수",
    color="species_korean",
    barmode="stack",
    text="개체 수",
    title="섬별 펭귄 종 분포",
    labels={
        "island_korean": "서식하는 섬",
        "species_korean": "펭귄 종"
    }
)

st.plotly_chart(fig_island, use_container_width=True)

st.subheader("섬별 종 분포표")

island_table = pd.crosstab(
    df["island_korean"],
    df["species_korean"]
)

st.dataframe(island_table, use_container_width=True)

st.info(
    "이 데이터에서는 아델리펭귄이 세 섬에서 모두 관찰되지만, "
    "턱끈펭귄은 드림섬, 젠투펭귄은 비스코섬에서만 관찰됩니다."
)


# ==========================================
# 8. 섬별 부리 길이와 깊이 분석
# ==========================================

st.header("6. 섬별 부리 형태 비교")

st.markdown(
    "펭귄이 관찰된 섬에 따라 부리 길이와 깊이가 "
    "어떻게 다른지 살펴봅니다."
)

col1, col2 = st.columns(2)

with col1:
    fig_island_length = px.box(
        df,
        x="island_korean",
        y="bill_length_mm",
        color="species_korean",
        points="all",
        title="섬별 부리 길이 비교",
        labels={
            "island_korean": "섬",
            "bill_length_mm": "부리 길이(mm)",
            "species_korean": "펭귄 종"
        }
    )
    st.plotly_chart(
        fig_island_length,
        use_container_width=True
    )

with col2:
    fig_island_depth = px.box(
        df,
        x="island_korean",
        y="bill_depth_mm",
        color="species_korean",
        points="all",
        title="섬별 부리 깊이 비교",
        labels={
            "island_korean": "섬",
            "bill_depth_mm": "부리 깊이(mm)",
            "species_korean": "펭귄 종"
        }
    )
    st.plotly_chart(
        fig_island_depth,
        use_container_width=True
    )

st.subheader("섬별 평균 부리 크기")

island_average = (
    df.groupby("island_korean")
    .agg(
        평균 부리 길이=("bill_length_mm", "mean"),
        평균 부리 깊이=("bill_depth_mm", "mean"),
        개체 수=("species", "count")
    )
    .round(2)
)

st.dataframe(island_average, use_container_width=True)

st.warning(
    "섬별 평균에는 서로 다른 펭귄 종이 섞여 있습니다. "
    "따라서 이 결과만으로 섬이 부리 형태에 영향을 준다고 "
    "판단할 수는 없습니다."
)


# ==========================================
# 9. 부리 길이와 깊이의 관계
# ==========================================

st.header("7. 부리 길이와 깊이의 관계")

fig_scatter = px.scatter(
    df,
    x="bill_length_mm",
    y="bill_depth_mm",
    color="species_korean",
    symbol="island_korean",
    hover_data=["island_korean", "species_korean"],
    title="펭귄 종과 섬에 따른 부리 형태 분포",
    labels={
        "bill_length_mm": "부리 길이(mm)",
        "bill_depth_mm": "부리 깊이(mm)",
        "species_korean": "펭귄 종",
        "island_korean": "서식하는 섬"
    }
)

st.plotly_chart(fig_scatter, use_container_width=True)

correlation = df[
    ["bill_length_mm", "bill_depth_mm"]
].corr().iloc[0, 1]

st.metric(
    "전체 펭귄의 부리 길이와 깊이 상관계수",
    round(correlation, 3)
)

st.caption(
    "상관계수는 두 수치가 함께 변하는 경향을 나타냅니다. "
    "인과관계를 의미하지는 않습니다."
)


# ==========================================
# 10. 같은 종 안에서 섬별 부리 형태 비교
# ==========================================

st.header("8. 같은 종 안에서 섬별 부리 형태 비교")

st.markdown(
    "종의 차이와 섬의 차이를 구분하기 위해 "
    "여러 섬에서 관찰된 동일한 종을 비교합니다."
)

species_island_counts = (
    df.groupby("species")["island"].nunique()
)

comparable_species = [
    species for species in SPECIES_NAMES
    if species_island_counts.get(species, 0) >= 2
]

if comparable_species:

    selected_species = st.selectbox(
        "비교할 펭귄 종 선택",
        comparable_species,
        format_func=lambda x: SPECIES_NAMES[x]
    )

    same_species_df = df[
        df["species"] == selected_species
    ].copy()

    col1, col2 = st.columns(2)

    with col1:
        fig_same_length = px.box(
            same_species_df,
            x="island_korean",
            y="bill_length_mm",
            color="island_korean",
            points="all",
            title="같은 종의 섬별 부리 길이",
            labels={
                "island_korean": "섬",
                "bill_length_mm": "부리 길이(mm)"
            }
        )
        st.plotly_chart(
            fig_same_length,
            use_container_width=True
        )

    with col2:
        fig_same_depth = px.box(
            same_species_df,
            x="island_korean",
            y="bill_depth_mm",
            color="island_korean",
            points="all",
            title="같은 종의 섬별 부리 깊이",
            labels={
                "island_korean": "섬",
                "bill_depth_mm": "부리 깊이(mm)"
            }
        )
        st.plotly_chart(
            fig_same_depth,
            use_container_width=True
        )

    same_species_average = (
        same_species_df.groupby("island_korean")
        .agg(
            평균 부리 길이=("bill_length_mm", "mean"),
            평균 부리 깊이=("bill_depth_mm", "mean"),
            개체 수=("species", "count")
        )
        .round(2)
    )

    st.subheader("동일 종의 섬별 평균")

    st.dataframe(
        same_species_average,
        use_container_width=True
    )

    st.caption(
        "여러 섬에서 관찰되는 종을 선택해 섬별 부리 차이를 비교합니다."
    )

else:
    st.info("여러 섬에서 관찰된 동일한 종이 없습니다.")


# ==========================================
# 11. 종별 부리 형태와 섬의 관계
# ==========================================

st.header("9. 종별 부리 형태의 평균 비교")

species_island_average = (
    df.groupby(["species_korean", "island_korean"])
    .agg(
        평균 부리 길이=("bill_length_mm", "mean"),
        평균 부리 깊이=("bill_depth_mm", "mean"),
        개체 수=("species", "count")
    )
    .round(2)
    .reset_index()
)

st.dataframe(
    species_island_average,
    use_container_width=True
)

fig_average = px.scatter(
    species_island_average,
    x="평균 부리 길이",
    y="평균 부리 깊이",
    color="species_korean",
    symbol="island_korean",
    size="개체 수",
    text="island_korean",
    title="종과 섬에 따른 평균 부리 형태",
    labels={
        "species_korean": "펭귄 종",
        "island_korean": "섬"
    }
)

st.plotly_chart(fig_average, use_container_width=True)


# ==========================================
# 12. 부리 형태를 이용한 종 예측
# ==========================================

st.header("10. 부리 길이와 깊이로 펭귄 종 예측")

st.markdown(
    "부리 길이와 깊이를 입력하면 기존 데이터에서 "
    "가장 가까운 부리 형태를 가진 펭귄을 찾아 종을 추정합니다."
)

col1, col2 = st.columns(2)

with col1:
    input_length = st.number_input(
        "부리 길이(mm)",
        min_value=25.0,
        max_value=70.0,
        value=45.0,
        step=0.1
    )

with col2:
    input_depth = st.number_input(
        "부리 깊이(mm)",
        min_value=10.0,
        max_value=25.0,
        value=17.0,
        step=0.1
    )

features = ["bill_length_mm", "bill_depth_mm"]

means = df[features].mean()
stds = df[features].std().replace(0, 1)

standardized_df = (df[features] - means) / stds

input_values = pd.Series({
    "bill_length_mm": input_length,
    "bill_depth_mm": input_depth
})

standardized_input = (input_values - means) / stds

distances = np.sqrt(
    ((standardized_df - standardized_input) ** 2).sum(axis=1)
)

nearest_index = distances.idxmin()
nearest_penguin = df.loc[nearest_index]

st.subheader("예측 결과")

st.success(
    f"가장 가까운 부리 형태는 {nearest_penguin['species_korean']}입니다."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("추정 종", nearest_penguin["species_korean"])

with col2:
    st.metric(
        "가장 가까운 데이터의 부리 길이",
        f"{nearest_penguin['bill_length_mm']:.1f} mm"
    )

with col3:
    st.metric(
        "가장 가까운 데이터의 부리 깊이",
        f"{nearest_penguin['bill_depth_mm']:.1f} mm"
    )

st.write(
    f"해당 데이터의 관찰 섬: "
    f"{nearest_penguin['island_korean']}"
)

st.caption(
    "이 기능은 학습된 머신러닝 모델이 아니라 "
    "표준화된 두 수치의 거리를 이용한 최근접 이웃 방식입니다. "
    "실제 종을 확정하는 결과는 아닙니다."
)

prediction_plot = px.scatter(
    df,
    x="bill_length_mm",
    y="bill_depth_mm",
    color="species_korean",
    symbol="island_korean",
    title="기존 펭귄 데이터와 입력한 부리 형태 비교",
    labels={
        "bill_length_mm": "부리 길이(mm)",
        "bill_depth_mm": "부리 깊이(mm)"
    }
)

prediction_plot.add_scatter(
    x=[input_length],
    y=[input_depth],
    mode="markers",
    marker=dict(
        size=18,
        color="red",
        symbol="star",
        line=dict(width=2, color="black")
    ),
    name="입력한 부리 형태"
)

st.plotly_chart(
    prediction_plot,
    use_container_width=True
)


# ==========================================
# 13. 전체 전처리 데이터
# ==========================================

st.header("11. 전체 데이터 확인")

with st.expander("전처리된 데이터 전체 보기"):
    st.dataframe(df, use_container_width=True)


# ==========================================
# 14. 데이터 출처
# ==========================================

st.header("12. 데이터 출처")

st.markdown("""
- 데이터 이름: Palmer Penguins
- 데이터 저장소: [GitHub - palmerpenguins](https://github.com/allisonhorst/palmerpenguins)
- 주요 분석 항목: 펭귄 종, 부리 길이, 부리 깊이, 서식 섬
- 데이터 출처: Palmer Station Long Term Ecological Research Program

이 사이트는 펭귄의 부리 형태와 종, 서식지 사이의 관계를
데이터를 통해 탐구하기 위해 제작되었습니다.
""")

st.caption(
    "주의: 관찰된 섬과 부리 형태 사이의 관계는 상관관계이며, "
    "섬이 부리 형태를 변화시켰다는 인과관계를 증명하지 않습니다."
)
