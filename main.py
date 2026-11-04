
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ==================================================
# 1. 기본 설정
# ==================================================

st.set_page_config(
    page_title="펭귄 부리 연구소",
    page_icon="🐧",
    layout="wide"
)

st.title("🐧 펭귄 부리 연구소")
st.markdown("""
펭귄의 **부리 길이와 깊이**를 중심으로 종과 서식지의 관계를 분석합니다.

Palmer Penguins 데이터를 이용해 펭귄의 형태적 특징과
섬별 분포에 어떤 관계가 있는지 탐구합니다.
""")

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


# ==================================================
# 2. 데이터 불러오기
# ==================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


try:
    raw_df = load_data()
except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.code(str(e))
    st.stop()


# ==================================================
# 3. 데이터 기본 정보
# ==================================================

st.header("1. 데이터 기본 정보")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("전체 데이터 행", f"{raw_df.shape[0]}개")

with col2:
    st.metric("전체 데이터 열", f"{raw_df.shape[1]}개")

with col3:
    st.metric("전체 결측 셀 수", f"{int(raw_df.isna().sum().sum())}개")

st.subheader("원본 데이터 미리보기")
st.dataframe(raw_df.head(10), use_container_width=True)

st.subheader("열별 데이터 유형")
dtype_table = raw_df.dtypes.astype(str).rename("데이터 유형")
st.dataframe(dtype_table, use_container_width=True)


# ==================================================
# 4. 결측치 분석
# ==================================================

st.header("2. 결측치 분석")

missing_summary = pd.DataFrame({
    "결측치 개수": raw_df.isna().sum(),
    "결측치 비율(%)": (raw_df.isna().mean() * 100).round(1)
})

st.dataframe(missing_summary, use_container_width=True)

missing_chart = missing_summary.reset_index()
missing_chart.columns = ["열 이름", "결측치 개수", "결측치 비율(%)"]

fig_missing = px.bar(
    missing_chart,
    x="열 이름",
    y="결측치 개수",
    text="결측치 개수",
    title="열별 결측치 개수",
    labels={"열 이름": "데이터 항목"}
)

st.plotly_chart(fig_missing, use_container_width=True)

st.info("""
원본 데이터는 344행, 8열입니다.
부리 길이와 깊이 등 일부 측정값과 성별 정보에 결측치가 있습니다.
결측치 개수는 비어 있는 셀의 수이며, 결측치가 있는 행의 수와는 다릅니다.
""")


# ==================================================
# 5. 데이터 전처리
# ==================================================

st.header("3. 데이터 전처리")

before_rows = len(raw_df)

# 분석에 필요한 값이 없는 행 제거
df = raw_df.dropna(
    subset=[
        "bill_length_mm",
        "bill_depth_mm",
        "species",
        "island"
    ]
).copy()

after_rows = len(df)
removed_rows = before_rows - after_rows

df["species_korean"] = df["species"].map(SPECIES_NAMES)
df["island_korean"] = df["island"].map(ISLAND_NAMES)

# 부리 길이 / 부리 깊이 비율
df["bill_ratio"] = (
    df["bill_length_mm"] / df["bill_depth_mm"]
).round(2)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("전처리 전", f"{before_rows}개")

with col2:
    st.metric("전처리 후", f"{after_rows}개")

with col3:
    st.metric("제외된 행", f"{removed_rows}개")

st.caption("""
부리 길이, 부리 깊이, 종, 섬 정보가 없는 행만 제외했습니다.
성별 등 이번 분석에 사용하지 않는 항목의 결측치는 유지했습니다.
""")


# ==================================================
# 6. 펭귄 종 분포
# ==================================================

st.header("4. 펭귄 종 분포 분석")

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


# ==================================================
# 7. 종별 부리 형태 분석
# ==================================================

st.header("5. 종별 부리 형태 분석")

col1, col2 = st.columns(2)

with col1:
    fig_length = px.box(
        df,
        x="species_korean",
        y="bill_length_mm",
        color="species_korean",
        points="all",
        title="종별 부리 길이 비교",
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
        title="종별 부리 깊이 비교",
        labels={
            "species_korean": "펭귄 종",
            "bill_depth_mm": "부리 깊이(mm)"
        }
    )
    st.plotly_chart(fig_depth, use_container_width=True)

species_average = (
    df.groupby("species_korean")
    .agg(
        평균 부리 길이=("bill_length_mm", "mean"),
        평균 부리 깊이=("bill_depth_mm", "mean"),
        평균 부리 비율=("bill_ratio", "mean"),
        개체 수=("species", "count")
    )
    .round(2)
)

st.subheader("종별 평균 부리 크기")
st.dataframe(species_average, use_container_width=True)


# ==================================================
# 8. 섬별 펭귄 종 분포
# ==================================================

st.header("6. 섬에 따른 펭귄 종 분포")

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
        "island_korean": "서식 섬",
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

st.info("""
데이터에서 아델리펭귄은 세 섬 모두에서 관찰됩니다.
턱끈펭귄은 드림섬, 젠투펭귄은 비스코섬에서 관찰됩니다.
따라서 종과 섬의 분포가 서로 밀접하게 연결되어 있습니다.
""")


# ==================================================
# 9. 섬별 부리 형태 분석
# ==================================================

st.header("7. 섬별 부리 형태 비교")

st.markdown("""
펭귄이 관찰된 섬에 따라 부리 길이와 깊이가 어떻게 다른지 비교합니다.
그래프의 색은 펭귄 종을 나타냅니다.
""")

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
    st.plotly_chart(fig_island_length, use_container_width=True)

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
    st.plotly_chart(fig_island_depth, use_container_width=True)

island_average = (
    df.groupby("island_korean")
    .agg(
        평균 부리 길이=("bill_length_mm", "mean"),
        평균 부리 깊이=("bill_depth_mm", "mean"),
        개체 수=("species", "count")
    )
    .round(2)
)

st.subheader("섬별 평균 부리 크기")
st.dataframe(island_average, use_container_width=True)

st.warning("""
섬별 평균은 서로 다른 종의 데이터가 섞인 값입니다.
따라서 섬별 평균 차이가 곧 서식지가 부리 형태에 영향을 주었다는 뜻은 아닙니다.
""")


# ==================================================
# 10. 부리 길이와 깊이의 관계
# ==================================================

st.header("8. 부리 형태 분포 지도")

fig_scatter = px.scatter(
    df,
    x="bill_length_mm",
    y="bill_depth_mm",
    color="species_korean",
    symbol="island_korean",
    hover_data=[
        "species_korean",
        "island_korean",
        "bill_ratio"
    ],
    title="종과 섬에 따른 부리 형태 분포",
    labels={
        "bill_length_mm": "부리 길이(mm)",
        "bill_depth_mm": "부리 깊이(mm)",
        "species_korean": "펭귄 종",
        "island_korean": "서식 섬"
    }
)

st.plotly_chart(fig_scatter, use_container_width=True)


# ==================================================
# 11. 같은 종 안에서 섬별 비교
# ==================================================

st.header("9. 같은 종 안에서 섬별 부리 형태 비교")

st.markdown("""
종에 따른 차이와 섬에 따른 차이를 구분하기 위해
여러 섬에서 관찰된 동일한 종을 비교합니다.
""")

species_island_counts = df.groupby("species")["island"].nunique()

comparable_species = [
    species
    for species in SPECIES_NAMES
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
            title="동일 종의 섬별 부리 길이",
            labels={
                "island_korean": "섬",
                "bill_length_mm": "부리 길이(mm)"
            }
        )
        st.plotly_chart(fig_same_length, use_container_width=True)

    with col2:
        fig_same_depth = px.box(
            same_species_df,
            x="island_korean",
            y="bill_depth_mm",
            color="island_korean",
            points="all",
            title="동일 종의 섬별 부리 깊이",
            labels={
                "island_korean": "섬",
                "bill_depth_mm": "부리 깊이(mm)"
            }
        )
        st.plotly_chart(fig_same_depth, use_container_width=True)

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
    st.dataframe(same_species_average, use_container_width=True)

    st.caption("""
    현재 데이터에서 여러 섬에 걸쳐 관찰된 종은 아델리펭귄입니다.
    따라서 이 비교는 아델리펭귄을 중심으로 해석해야 합니다.
    """)

else:
    st.info("여러 섬에서 관찰된 동일한 종이 없습니다.")


# ==================================================
# 12. 상관관계 분석
# ==================================================

st.header("10. 부리 길이와 깊이의 상관관계")

# 전체 데이터 상관계수
overall_corr = df[
    ["bill_length_mm", "bill_depth_mm"]
].corr().iloc[0, 1]

st.metric(
    "전체 펭귄의 상관계수",
    f"{overall_corr:.3f}"
)

# 종별 상관계수
corr_by_species = (
    df.groupby("species_korean")
    .apply(
        lambda group: group[
            ["bill_length_mm", "bill_depth_mm"]
        ].corr().iloc[0, 1],
        include_groups=False
    )
    .reset_index(name="상관계수")
)

corr_by_species["상관계수"] = corr_by_species["상관계수"].round(3)

st.subheader("종별 부리 길이와 깊이의 상관관계")
st.dataframe(corr_by_species, use_container_width=True)

fig_corr = px.bar(
    corr_by_species,
    x="species_korean",
    y="상관계수",
    color="species_korean",
    text="상관계수",
    title="펭귄 종별 부리 길이와 깊이의 상관관계",
    labels={
        "species_korean": "펭귄 종",
        "상관계수": "상관계수"
    }
)

st.plotly_chart(fig_corr, use_container_width=True)

st.info("""
전체 펭귄을 한꺼번에 분석한 결과와 종별로 나누어 분석한 결과는
서로 다를 수 있습니다. 종마다 부리 형태의 평균이 다르기 때문입니다.
이처럼 집단을 구분했을 때 전체적인 경향이 달라지는 현상은
집단 구성의 영향을 받는 통계적 현상과 관련이 있습니다.
""")

st.caption("""
상관관계는 두 수치가 함께 변하는 경향을 나타냅니다.
한쪽이 다른 쪽의 원인이라는 뜻은 아닙니다.
""")


# ==================================================
# 13. 이상치 분석
# ==================================================

st.header("11. 부리 데이터 이상치 분석")

st.markdown("""
IQR 방식을 이용해 부리 길이와 깊이에서
통계적으로 특이한 값을 확인합니다.
""")

outlier_columns = [
    "bill_length_mm",
    "bill_depth_mm"
]

outlier_results = []

for column in outlier_columns:

    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = df[
        (df[column] < lower) |
        (df[column] > upper)
    ]

    outlier_results.append({
        "항목": column,
        "하한": round(lower, 2),
        "상한": round(upper, 2),
        "이상치 후보 개수": len(outliers)
    })

outlier_summary = pd.DataFrame(outlier_results)

st.subheader("전체 데이터 기준 이상치 후보")
st.dataframe(outlier_summary, use_container_width=True)

# 종별 IQR 분석
species_outlier_results = []

for species, group in df.groupby("species_korean"):

    for column in outlier_columns:

        q1 = group[column].quantile(0.25)
        q3 = group[column].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        count = (
            (group[column] < lower) |
            (group[column] > upper)
        ).sum()

        species_outlier_results.append({
            "펭귄 종": species,
            "측정 항목": column,
            "이상치 후보 개수": int(count)
        })

st.subheader("종별 이상치 후보")
st.dataframe(
    pd.DataFrame(species_outlier_results),
    use_container_width=True
)

# 이상치 후보 상세 확인
for column in outlier_columns:

    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = df[
        (df[column] < lower) |
        (df[column] > upper)
    ]

    with st.expander(f"{column} 이상치 후보 상세 보기"):

        if len(outliers) > 0:
            st.dataframe(
                outliers[
                    [
                        "species_korean",
                        "island_korean",
                        column
                    ]
                ],
                use_container_width=True
            )
        else:
            st.write("이상치 후보가 없습니다.")

st.warning("""
이상치 후보가 반드시 측정 오류라는 뜻은 아닙니다.
종마다 부리 형태가 다르기 때문에 실제 생물학적 차이일 수도 있습니다.
이번 분석에서는 이상치 후보를 임의로 삭제하지 않았습니다.
""")


# ==================================================
# 14. 부리 길이와 깊이로 펭귄 종 추정
# ==================================================

st.header("12. 부리 길이와 깊이로 펭귄 종 추정")

st.markdown("""
부리 길이와 깊이를 입력하면 기존 데이터에서
가장 가까운 부리 형태를 가진 펭귄을 찾아 종을 추정합니다.
""")

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

features = [
    "bill_length_mm",
    "bill_depth_mm"
]

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

st.subheader("추정 결과")

st.success(
    f"가장 가까운 부리 형태는 "
    f"{nearest_penguin['species_korean']}입니다."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("추정 종", nearest_penguin["species_korean"])

with col2:
    st.metric(
        "가장 가까운 부리 길이",
        f"{nearest_penguin['bill_length_mm']:.1f} mm"
    )

with col3:
    st.metric(
        "가장 가까운 부리 깊이",
        f"{nearest_penguin['bill_depth_mm']:.1f} mm"
    )

st.write(
    f"해당 데이터의 관찰 섬: "
    f"{nearest_penguin['island_korean']}"
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
        "bill_depth_mm": "부리 깊이(mm)",
        "species_korean": "펭귄 종",
        "island_korean": "섬"
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

st.plotly_chart(prediction_plot, use_container_width=True)

st.caption("""
이 기능은 학습된 머신러닝 모델이 아니라
표준화된 부리 길이와 깊이의 거리를 이용한 최근접 이웃 방식입니다.
실제 펭귄의 종을 확정하는 결과는 아닙니다.
""")


# ==================================================
# 15. 전체 데이터 확인
# ==================================================

st.header("13. 전처리된 전체 데이터")

with st.expander("전처리된 데이터 전체 보기"):
    st.dataframe(df, use_container_width=True)


# ==================================================
# 16. 데이터 출처 및 분석 한계
# ==================================================

st.header("14. 데이터 출처와 분석 한계")

st.markdown("""
**데이터 이름:** Palmer Penguins

**데이터 저장소:** [GitHub - palmerpenguins](https://github.com/allisonhorst/palmerpenguins)

**주요 분석 항목**
- 펭귄 종
- 부리 길이(mm)
- 부리 깊이(mm)
- 서식 섬
- 날개 길이와 몸무게 등 원본 데이터 항목

**분석 시 주의할 점**
1. 섬별 종 분포가 서로 다르므로 섬별 평균만으로 서식지의 영향을 단정할 수 없습니다.
2. 부리 길이와 깊이의 상관관계가 인과관계를 의미하지는 않습니다.
3. IQR 방식으로 찾은 이상치 후보는 실제 측정 오류라고 확정할 수 없습니다.
4. 종 예측 기능은 최근접 이웃 방식의 간단한 추정이며, 학습된 머신러닝 모델이 아닙니다.
5. 이 데이터는 세 종의 펭귄을 대상으로 하므로 모든 펭귄 종에 일반화할 수 없습니다.
""")

st.success("펭귄 부리 연구소 분석이 완료되었습니다!")
