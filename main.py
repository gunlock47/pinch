
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Beak Finder",
    page_icon="🐧",
    layout="wide"
)

DATA_URL = (
    "https://raw.githubusercontent.com/allisonhorst/"
    "palmerpenguins/main/inst/extdata/penguins.csv"
)

SPECIES_KO = {
    "Adelie": "아델리펭귄",
    "Chinstrap": "턱끈펭귄",
    "Gentoo": "젠투펭귄"
}

SPECIES_INFO = {
    "Adelie": "남극 주변에 서식하며 부리 길이와 깊이를 통해 특징을 비교할 수 있습니다.",
    "Chinstrap": "턱 아래 검은 띠가 특징이며 다른 펭귄과 부리 측정값을 비교할 수 있습니다.",
    "Gentoo": "주황빛 부리가 특징이며 세 종의 측정값을 비교할 수 있습니다."
}


@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv(DATA_URL)

    df = df.dropna(
        subset=[
            "species",
            "bill_length_mm",
            "bill_depth_mm"
        ]
    ).copy()

    df["종"] = df["species"].map(SPECIES_KO)
    df["부리길이"] = df["bill_length_mm"]
    df["부리깊이"] = df["bill_depth_mm"]

    return df


st.title("🐧 Beak Finder")
st.subheader("부리의 길이와 깊이로 펭귄 종류 찾기")

st.write(
    "부리 길이와 깊이를 입력하면 실제 펭귄 측정 데이터와 "
    "비교해 가장 비슷한 종을 찾아줍니다."
)

st.info(
    "이 사이트는 펭귄 3종의 측정값을 비교하는 교육용 도구입니다. "
    "부리 모양만으로 종을 확정할 수는 없습니다."
)

try:
    df = load_data()
except Exception:
    st.error(
        "데이터를 불러오지 못했습니다. 인터넷 연결을 확인해 주세요."
    )
    st.stop()


st.header("🔎 부리 측정값 입력")

col1, col2 = st.columns(2)

with col1:
    length = st.number_input(
        "부리 길이 (mm)",
        min_value=0.1,
        max_value=200.0,
        value=40.0,
        step=0.5
    )

with col2:
    depth = st.number_input(
        "부리 깊이 (mm)",
        min_value=0.1,
        max_value=100.0,
        value=18.0,
        step=0.5
    )


# 부리 길이와 깊이의 표준편차를 이용해 거리를 계산합니다.
length_std = df["bill_length_mm"].std()
depth_std = df["bill_depth_mm"].std()

if pd.isna(length_std) or length_std == 0:
    length_std = 1

if pd.isna(depth_std) or depth_std == 0:
    depth_std = 1

df["거리"] = (
    ((df["bill_length_mm"] - length) / length_std) ** 2
    + ((df["bill_depth_mm"] - depth) / depth_std) ** 2
) ** 0.5


# 입력한 값과 가장 가까운 펭귄 측정값을 찾습니다.
nearest = df.loc[df["거리"].idxmin()]
predicted_species = nearest["species"]


st.divider()
st.header("🐧 분석 결과")

st.subheader(
    f"가장 비슷한 종: {SPECIES_KO[predicted_species]}"
)

st.write(SPECIES_INFO[predicted_species])

result1, result2 = st.columns(2)

with result1:
    st.metric("입력한 부리 길이", f"{length:.2f} mm")
    st.metric(
        "비교 대상의 부리 길이",
        f"{nearest['bill_length_mm']:.2f} mm"
    )

with result2:
    st.metric("입력한 부리 깊이", f"{depth:.2f} mm")
    st.metric(
        "비교 대상의 부리 깊이",
        f"{nearest['bill_depth_mm']:.2f} mm"
    )

st.caption(
    "비교 대상의 측정값은 데이터에서 가장 가까운 개체의 값입니다. "
    "종 전체의 평균값과는 다를 수 있습니다."
)


st.header("📊 펭귄 종별 부리 측정값")

fig = px.scatter(
    df,
    x="bill_length_mm",
    y="bill_depth_mm",
    color="종",
    hover_data=["island", "sex"],
    labels={
        "bill_length_mm": "부리 길이 (mm)",
        "bill_depth_mm": "부리 깊이 (mm)",
        "종": "펭귄 종류"
    },
    title="펭귄의 부리 길이와 깊이 비교"
)

fig.add_scatter(
    x=[length],
    y=[depth],
    mode="markers",
    name="입력한 값",
    marker=dict(size=14, symbol="x")
)

st.plotly_chart(fig, use_container_width=True)


with st.expander("전체 측정 데이터 보기"):
    st.dataframe(
        df[
            [
                "종",
                "bill_length_mm",
                "bill_depth_mm",
                "island",
                "sex"
            ]
        ].rename(
            columns={
                "bill_length_mm": "부리 길이(mm)",
                "bill_depth_mm": "부리 깊이(mm)",
                "island": "조사 섬",
                "sex": "성별"
            }
        ),
        use_container_width=True
    )


st.divider()
st.subheader("데이터 출처")

st.markdown(
    "[Palmer Penguins 데이터셋](https://github.com/"
    "allisonhorst/palmerpenguins/blob/main/inst/extdata/penguins.csv)"
)

st.markdown(
    "[데이터 설명](https://allisonhorst.github.io/palmerpenguins/)"
)
