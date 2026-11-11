
# =========================================
# 9. ADVANCED DATA VISUALIZATION
# =========================================

st.divider()

st.markdown(
    '<div class="section-label">03 / ADVANCED DATA ANALYSIS</div>',
    unsafe_allow_html=True
)

st.header("📊 PENGUIN DATA ANALYSIS")
st.caption("8가지 시각화로 살펴보는 펭귄의 부리와 서식지")

# 그래프를 4개 그룹으로 나누기
tab1, tab2, tab3, tab4 = st.tabs([
    "🧬 부리 분석",
    "🌎 서식지 분석",
    "📈 데이터 분포",
    "🧠 AI 분석"
])

# -----------------------------------------
# TAB 1 : 부리 분석
# -----------------------------------------

with tab1:

    st.subheader("01. 종별 부리 크기 비교")

    col1, col2 = st.columns(2)

    with col1:

        fig = px.box(
            df,
            x="species",
            y="bill_length_mm",
            color="species",
            color_discrete_map=colors,
            template=plot_theme,
            points="all",
            title="종별 부리 길이 비교",
            labels={
                "species": "펭귄 종",
                "bill_length_mm": "부리 길이 (mm)"
            }
        )

        fig.update_layout(height=430)
        st.plotly_chart(fig, use_container_width=True)

    with col2:

        fig = px.box(
            df,
            x="species",
            y="bill_depth_mm",
            color="species",
            color_discrete_map=colors,
            template=plot_theme,
            points="all",
            title="종별 부리 깊이 비교",
            labels={
                "species": "펭귄 종",
                "bill_depth_mm": "부리 깊이 (mm)"
            }
        )

        fig.update_layout(height=430)
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "상자그림을 통해 종마다 부리 길이와 깊이의 "
        "중앙값 및 데이터 분포를 비교할 수 있습니다."
    )

    st.divider()

    st.subheader("02. 부리 길이와 깊이의 관계")

    fig = px.scatter(
        df,
        x="bill_length_mm",
        y="bill_depth_mm",
        color="species",
        color_discrete_map=colors,
        hover_data=["island"],
        template=plot_theme,
        title="펭귄 종별 부리 형태 산점도",
        labels={
            "bill_length_mm": "부리 길이 (mm)",
            "bill_depth_mm": "부리 깊이 (mm)",
            "species": "펭귄 종"
        }
    )

    fig.update_traces(
        marker=dict(size=11, opacity=0.85)
    )

    fig.update_layout(height=500)

    st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------
# TAB 2 : 서식지 분석
# -----------------------------------------

with tab2:

    st.subheader("03. 섬별 펭귄 개체 수")

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
        template=plot_theme,
        title="섬에 따른 펭귄 종별 개체 수",
        text_auto=True
    )

    fig.update_layout(height=450)

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("04. 섬별 펭귄 종 구성 비율")

    fig = px.bar(
        island_df,
        x="island",
        y="개체 수",
        color="species",
        color_discrete_map={
            korean[k]: colors[k] for k in colors
        },
        barmode="relative",
        barnorm="percent",
        template=plot_theme,
        title="각 섬에서 종별로 차지하는 비율",
        labels={
            "island": "서식 섬",
            "개체 수": "구성 비율 (%)"
        }
    )

    fig.update_layout(
        height=450,
        yaxis_title="구성 비율 (%)"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "각 섬의 전체 펭귄 중 어떤 종이 큰 비율을 차지하는지 확인합니다."
    )


# -----------------------------------------
# TAB 3 : 데이터 분포
# -----------------------------------------

with tab3:

    st.subheader("05. 부리 길이 분포")

    fig = px.histogram(
        df,
        x="bill_length_mm",
        color="species",
        color_discrete_map=colors,
        template=plot_theme,
        barmode="overlay",
        opacity=0.75,
        nbins=25,
        title="펭귄 종별 부리 길이 분포",
        labels={
            "bill_length_mm": "부리 길이 (mm)",
            "count": "펭귄 개체 수",
            "species": "펭귄 종"
        }
    )

    fig.update_layout(height=450)

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("06. 변수 간 상관관계 히트맵")

    numeric_data = df[
        [
            "bill_length_mm",
            "bill_depth_mm",
            "flipper_length_mm",
            "body_mass_g"
        ]
    ].dropna()

    numeric_data = numeric_data.rename(
        columns={
            "bill_length_mm": "부리 길이",
            "bill_depth_mm": "부리 깊이",
            "flipper_length_mm": "날개 길이",
            "body_mass_g": "몸무게"
        }
    )

    corr = numeric_data.corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale=[
            "#6650A4",
            "#172642",
            "#35D6E8"
        ],
        zmin=-1,
        zmax=1,
        template=plot_theme,
        title="펭귄 신체 특징의 상관관계"
    )

    fig.update_layout(height=500)

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "1에 가까우면 양의 관계, -1에 가까우면 음의 관계이며, "
        "0에 가까울수록 선형 관계가 약합니다."
    )


# -----------------------------------------
# TAB 4 : AI 분석
# -----------------------------------------

with tab4:

    st.subheader("07. AI 모델 변수 영향 비교")

    # 모델에 학습된 로지스틱 회귀 계수
    classifier = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessing"]

    feature_names = preprocessor.get_feature_names_out()

    importance = abs(classifier.coef_).mean(axis=0)

    importance_df = pd.DataFrame({
        "변수": feature_names,
        "영향 크기": importance
    })

    importance_df["변수"] = (
        importance_df["변수"]
        .str.replace("numeric__", "", regex=False)
        .str.replace("island__island_", "섬: ", regex=False)
    )

    importance_df = importance_df.sort_values(
        "영향 크기",
        ascending=True
    )

    fig = px.bar(
        importance_df,
        x="영향 크기",
        y="변수",
        orientation="h",
        color="영향 크기",
        color_continuous_scale=[
            "#6650A4",
            "#267DAB",
            "#35D6E8"
        ],
        template=plot_theme,
        title="분류 모델의 변수별 계수 크기",
        labels={
            "영향 크기": "평균 절댓값 계수",
            "변수": "분류에 사용된 변수"
        }
    )

    fig.update_layout(height=450)

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "로지스틱 회귀 계수의 절댓값을 종별로 평균한 값입니다. "
        "인과관계나 독립적인 변수 중요도를 직접 의미하지는 않습니다."
    )

    st.divider()

    st.subheader("08. AI 분류 결과")

    fig = px.imshow(
        matrix,
        x=[korean[x] for x in model.classes_],
        y=[korean[x] for x in model.classes_],
        text_auto=True,
        color_continuous_scale=[
            "#111C31",
            "#176B87",
            "#35D6E8"
        ],
        template=plot_theme,
        title="실제 종과 AI 예측 종 비교",
        labels={
            "x": "예측한 종",
            "y": "실제 종",
            "color": "개체 수"
        }
    )

    fig.update_layout(height=450)

    st.plotly_chart(fig, use_container_width=True)

    st.metric(
        "AI 시험 데이터 정확도",
        f"{accuracy * 100:.2f}%"
    )


# -----------------------------------------
# DATA FOOTNOTE
# -----------------------------------------

st.markdown("""
<div style="
    background:#111C31;
    border:1px solid #304D70;
    padding:20px;
    border-radius:15px;
    text-align:center;
">
    <span style="color:#5DE4F5; letter-spacing:3px;">
        PENGUIN DATA SCIENCE LAB
    </span>
    <p style="color:#91A7C7; font-size:12px;">
        8 VISUALIZATION MODULES · PALMER PENGUINS
    </p>
</div>
""", unsafe_allow_html=True)
