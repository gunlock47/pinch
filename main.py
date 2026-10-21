
import streamlit as st
import pandas as pd
import plotly.express as px
import requests

st.set_page_config(
    page_title="Beak Finder",
    page_icon="🐦",
    layout="wide"
)

DATA_URL = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv"

SPECIES_KO = {
    "Adelie": "아델리펭귄",
    "Chinstrap": "턱끈펭귄",
    "Gentoo": "젠투펭귄"
}

SPECIES_INFO = {
    "Adelie": "비교적 짧고 깊은 부리가 특징인 펭귄입니다.",
    "Chinstrap": "상대적으로 길고 가는 부리를 가진 펭귄입니다.",
    "Gentoo": "주황빛 부리와 큰 몸집이 특징인 펭귄입니다."
}

WIKI_TITLES = {
    "Adelie": "Adélie_penguin",
    "Chinstrap": "Chinstrap_penguin",
    "Gentoo": "Gentoo_penguin"
}


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)
    df = df.dropna(
        subset=["species", "bill_length_mm", "bill_depth_mm"]
    )
    df["species_ko"] = df["species"].map(SPECIES_KO)
    return df


@st.cache_data(ttl=86400)
def get_image(title):
    try:
        url = (
            "https://en.wikipedia.org/api/rest_v1/page/summary/"
            + requests.utils.quote(title, safe="_")
        )
        response = requests.get(url, timeout=8)
        response.raise_for_status()
        return response.json().get("thumbnail", {}).get("source")
    except Exception:
        return None


st.title("🐦 Beak Finder")
st.subheader("부리 모양으로 펭귄 찾아보기")
st.write(
    "부리의 길이와 깊이를 선택하면 조건에 맞는 펭귄 종을 확인할 수 있습니다."
)

st.info(
    "이 사이트는 Palmer Penguins 공개 데이터를 이용합니다. "
    "현재 아델리펭귄, 턱끈펭귄, 젠투펭귄 3종을 대상으로 합니다."
)

try:
    df = load_data()
except Exception:
    st.error("데이터를 불러오지 못했습니다. 잠시 후 다시 시도해 주세요.")
    st.stop()

st.sidebar.header("부리 유형 선택")

beak_type = st.sidebar.selectbox(
    "찾고 싶은 부리 유형",
    [
        "전체 보기",
        "길고 얕은 부리",
        "길고 깊은 부리",
        "짧고 얕은 부리",
        "짧고 깊은 부리"
    ]
)

length_mid = df["bill_length_mm"].median()
depth_mid = df["bill_depth_mm"].median()

if beak_type == "길고 얕은 부리":
    result = df[
        (df["bill_length_mm"] >= length_mid)
        & (df["bill_depth_mm"] < depth_mid)
    ]
elif beak_type == "길고 깊은 부리":
    result = df[
        (df["bill_length_mm"] >= length_mid)
        & (df["bill_depth_mm"] >= depth_mid)
    ]
elif beak_type == "짧고 얕은 부리":
    result = df[
        (df["bill_length_mm"] < length_mid)
        & (df["bill_depth_mm"] < depth_mid)
    ]
elif beak_type == "짧고 깊은 부리":
    result = df[
        (df["bill_length_mm"] < length_mid)
        & (df["bill_depth_mm"] >= depth_mid)
    ]
else:
    result = df.copy()

st.caption(
    f"분석 데이터: {len(df)}개 관측값 | "
    "부리 유형은 전체 데이터의 중앙값을 기준으로 구분합니다."
)

st.header("🔎 검색 결과")

if result.empty:
    st.warning("해당 부리 유형의 데이터가 없습니다.")
    st.stop()

counts = result["species"].value_counts()

species_order = ["Adelie", "Chinstrap", "Gentoo"]
found_species = [s for s in species_order if s in counts.index]

cols = st.columns(len(found_species))

for col, species in zip(cols, found_species):
    with col:
        st.subheader(SPECIES_KO[species])

        image_url = get_image(WIKI_TITLES[species])

        if image_url:
            st.image(image_url, use_container_width=True)
        else:
            st.markdown("### 🐧")
            st.caption("사진을 불러올 수 없습니다.")

        st.write(SPECIES_INFO[species])
        st.metric(
            "조건에 해당하는 기록",
            f"{int(counts[species])}개"
        )

st.divider()

st.header("📊 부리 크기 비교")

plot_data = result.copy()
plot_data["종"] = plot_data["species"].map(SPECIES_KO)

fig = px.scatter(
    plot_data,
    x="bill_length_mm",
    y="bill_depth_mm",
    color="종",
    hover_data=["island", "year"],
    labels={
        "bill_length_mm": "부리 길이 (mm)",
        "bill_depth_mm": "부리 깊이 (mm)",
        "종": "펭귄 종"
    },
    title="부리 길이와 깊이의 관계"
)

st.plotly_chart(fig, use_container_width=True)

with st.expander("측정 데이터 확인하기"):
    st.dataframe(
        result[
            [
                "species_ko",
                "bill_length_mm",
                "bill_depth_mm",
                "flipper_length_mm",
                "body_mass_g",
                "island"
            ]
        ].rename(columns={
            "species_ko": "펭귄 종",
            "bill_length_mm": "부리 길이(mm)",
            "bill_depth_mm": "부리 깊이(mm)",
            "flipper_length_mm": "날개 길이(mm)",
            "body_mass_g": "몸무게(g)",
            "island": "조사 지역"
        }),
        use_container_width=True
    )

st.divider()

st.header("📚 데이터 출처")

st.markdown("""
- [Palmer Penguins 데이터 설명](https://allisonhorst.github.io/palmerpenguins/)
- [원본 CSV 데이터](https://github.com/allisonhorst/palmerpenguins/blob/main/inst/extdata/penguins.csv)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/690/palmer+penguins-3)

이 사이트는 교육용 데이터 탐색 도구입니다.
부리 형태만으로 모든 새의 종을 확정할 수는 없습니다.
""")
