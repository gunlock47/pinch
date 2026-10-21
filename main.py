
import io
import zipfile

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

st.set_page_config(
    page_title="Beak Finder | 새의 부리 탐구",
    page_icon="🐦",
    layout="wide",
)

PENGUIN_URL = (
    "https://raw.githubusercontent.com/allisonhorst/"
    "palmerpenguins/main/inst/extdata/penguins.csv"
)

SOURCES = {
    "펭귄": "https://allisonhorst.github.io/palmerpenguins/",
    "물새": "https://datadryad.org/dataset/doi:10.5061/dryad.42s0c",
    "카나리아": "https://datadryad.org/dataset/doi:10.5061/dryad.d51c5b0dm",
}

st.title("🐦 Beak Finder")
st.subheader("부리의 특징으로 새를 탐구해 보세요")
st.write(
    "펭귄, 물새, 카나리아의 공개 데이터를 활용해 "
    "부리의 크기와 형태, 먹이와의 관계를 살펴보는 생명과학 데이터 웹사이트입니다."
)

st.info(
    "펭귄 데이터는 자동으로 불러옵니다. "
    "물새와 카나리아는 각 데이터 출처에서 받은 파일을 업로드하면 분석할 수 있습니다."
)


@st.cache_data(ttl=3600)
def load_penguins():
    response = requests.get(PENGUIN_URL, timeout=20)
    response.raise_for_status()
    df = pd.read_csv(io.BytesIO(response.content))
    return df.dropna(
        subset=["species", "bill_length_mm", "bill_depth_mm"]
    )


def read_uploaded_file(uploaded_file):
    """CSV, TXT, Excel 파일을 읽습니다."""
    name = uploaded_file.name.lower()
    content = uploaded_file.getvalue()

    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(content))
    if name.endswith(".txt"):
        return pd.read_csv(
            io.BytesIO(content),
            sep=None,
            engine="python",
            comment="#",
        )
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(content))

    raise ValueError("CSV, TXT, XLSX, XLS 파일을 업로드해 주세요.")


def show_dataset(df, title):
    """데이터 미리보기와 숫자형 자료의 그래프를 표시합니다."""
    st.subheader(title)
    st.write(f"데이터 행 수: **{len(df):,}개**")
    st.dataframe(df.head(100), use_container_width=True)

    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] >= 1 and len(numeric) > 0:
        columns = numeric.columns.tolist()
        x_col = st.selectbox(
            f"{title} - 가로축",
            columns,
            key=f"{title}_x",
        )

        if len(columns) >= 2:
            y_options = [c for c in columns if c != x_col]
            y_col = st.selectbox(
                f"{title} - 세로축",
                y_options,
                key=f"{title}_y",
            )

            plot_df = df[[x_col, y_col]].dropna()
            if len(plot_df) > 0:
                fig = px.scatter(
                    plot_df,
                    x=x_col,
                    y=y_col,
                    title=f"{x_col}과 {y_col} 비교",
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.histogram(
                df,
                x=x_col,
                title=f"{x_col} 분포",
            )
            st.plotly_chart(fig, use_container_width=True)


tab1, tab2, tab3, tab4 = st.tabs(
    ["🐧 펭귄", "🦆 물새", "🐤 카나리아", "📊 데이터 비교"]
)

with tab1:
    st.header("펭귄 부리 분석")
    st.write(
        "아델리펭귄, 턱끈펭귄, 젠투펭귄의 부리 길이와 깊이를 비교합니다."
    )

    try:
        penguins = load_penguins()
        species_names = {
            "Adelie": "아델리펭귄",
            "Chinstrap": "턱끈펭귄",
            "Gentoo": "젠투펭귄",
        }
        penguins["종"] = penguins["species"].map(species_names)

        selected = st.multiselect(
            "비교할 펭귄 종류",
            options=list(species_names.values()),
            default=list(species_names.values()),
            key="penguin_species",
        )
        filtered = penguins[penguins["종"].isin(selected)]

        col1, col2, col3 = st.columns(3)
        col1.metric("전체 관측값", f"{len(penguins)}개")
        col2.metric("부리 길이 평균", f"{penguins['bill_length_mm'].mean():.1f} mm")
        col3.metric("부리 깊이 평균", f"{penguins['bill_depth_mm'].mean():.1f} mm")

        fig = px.scatter(
            filtered,
            x="bill_length_mm",
            y="bill_depth_mm",
            color="종",
            hover_data=["island", "sex"],
            labels={
                "bill_length_mm": "부리 길이 (mm)",
                "bill_depth_mm": "부리 깊이 (mm)",
            },
            title="펭귄 종별 부리 길이와 깊이",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            filtered[
                ["종", "bill_length_mm", "bill_depth_mm",
                 "flipper_length_mm", "body_mass_g"]
            ].rename(
                columns={
                    "bill_length_mm": "부리 길이(mm)",
                    "bill_depth_mm": "부리 깊이(mm)",
                    "flipper_length_mm": "날개 길이(mm)",
                    "body_mass_g": "몸무게(g)",
                }
            ),
            use_container_width=True,
        )

    except Exception as e:
        st.error(f"펭귄 데이터를 불러오지 못했습니다: {e}")

    st.markdown(f"데이터 출처: [Palmer Penguins]({SOURCES['펭귄']})")


with tab2:
    st.header("물새의 부리 형태와 먹이")
    st.write(
        "물새 42종을 대상으로 부리 형태와 먹이 구성의 관계를 연구한 데이터입니다. "
        "원본 자료에는 종별 부리 형태 좌표와 먹이 구성 자료가 포함됩니다."
    )

    st.markdown(f"원본 데이터: [Dryad 물새 데이터셋]({SOURCES['물새']})")
    st.caption(
        "추천 파일: Diet composition.txt 또는 Diet composition full.xls. "
        "부리 형태 좌표 파일과 먹이 구성 파일은 서로 다른 자료입니다."
    )

    waterfowl_file = st.file_uploader(
        "물새 데이터 파일 업로드",
        type=["csv", "txt", "xlsx", "xls"],
        key="waterfowl_file",
    )

    if waterfowl_file:
        try:
            waterfowl = read_uploaded_file(waterfowl_file)
            st.success("물새 데이터 파일을 읽었습니다.")
            show_dataset(waterfowl, "물새 데이터 미리보기")
        except Exception as e:
            st.error(f"파일을 읽지 못했습니다: {e}")
    else:
        st.info("물새 데이터 파일을 업로드하면 표와 그래프가 표시됩니다.")


with tab3:
    st.header("카나리아의 부리 측정")
    st.write(
        "카나리아 87개체를 조사한 연구 자료입니다. "
        "부리 길이, 높이(깊이), 너비 등의 측정값과 먹이 처리 성능을 분석할 수 있습니다."
    )

    st.markdown(f"원본 데이터: [Dryad 카나리아 데이터셋]({SOURCES['카나리아']})")
    st.caption(
        "추천 파일: shape_summary.csv. "
        "다운로드한 원본 파일에서 이 CSV를 선택해 업로드하세요."
    )

    canary_file = st.file_uploader(
        "카나리아 shape_summary.csv 업로드",
        type=["csv", "txt", "xlsx", "xls"],
        key="canary_file",
    )

    if canary_file:
        try:
            canary = read_uploaded_file(canary_file)
            st.success("카나리아 데이터 파일을 읽었습니다.")

            # 실제 데이터에 존재하는 부리 측정 열을 찾아 표시합니다.
            beak_columns = [
                col for col in canary.columns
                if any(
                    word in str(col).lower()
                    for word in [
                        "beak_length",
                        "beak_height",
                        "beak_width",
                        "upper_angle",
                        "lower_angle",
                    ]
                )
            ]

            if beak_columns:
                st.subheader("부리 측정값")
                st.dataframe(
                    canary[beak_columns].describe().T,
                    use_container_width=True,
                )

            show_dataset(canary, "카나리아 데이터 미리보기")

        except Exception as e:
            st.error(f"파일을 읽지 못했습니다: {e}")
    else:
        st.info("카나리아 CSV 파일을 업로드하면 측정값과 그래프가 표시됩니다.")


with tab4:
    st.header("세 종류의 새 데이터 비교")
    st.write(
        "세 데이터셋은 측정 방법과 변수가 서로 다르기 때문에 "
        "같은 기준으로 직접 비교할 수 있는 항목만 비교해야 합니다."
    )

    st.markdown("### 데이터셋 안내")
    comparison = pd.DataFrame(
        [
            {
                "새 종류": "펭귄",
                "자료 내용": "종별 부리 길이와 깊이",
                "분석 방법": "자동 로드 및 종별 산점도",
            },
            {
                "새 종류": "물새",
                "자료 내용": "종별 부리 형태 좌표와 먹이 구성",
                "분석 방법": "파일 업로드 후 표와 수치 그래프",
            },
            {
                "새 종류": "카나리아",
                "자료 내용": "개체별 부리 길이·높이·너비 등",
                "분석 방법": "파일 업로드 후 측정값과 그래프",
            },
        ]
    )
    st.dataframe(comparison, hide_index=True, use_container_width=True)

    st.warning(
        "물새의 형태 좌표와 펭귄·카나리아의 길이 측정값은 "
        "동일한 단위나 측정 기준이 아닐 수 있습니다. "
        "자료의 변수 설명을 확인한 뒤 비교해야 합니다."
    )


st.divider()
st.subheader("이 프로젝트에서 알아볼 수 있는 점")
st.markdown(
    """
    - 새의 종류에 따라 부리의 크기와 형태가 어떻게 다른지 탐구합니다.
    - 물새의 부리 형태와 먹이 구성 사이의 관계를 살펴봅니다.
    - 카나리아 개체 사이의 부리 측정값 차이를 분석합니다.
    - 공개 생명과학 데이터를 표와 그래프로 시각화합니다.
    """
)

st.caption(
    "참고: 이 웹사이트는 교육용 데이터 탐구 도구입니다. "
    "부리 모양만으로 모든 새의 종을 판별하는 인공지능 모델은 아닙니다."
)
