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
