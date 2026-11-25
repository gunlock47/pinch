from pathlib import Path

path = Path("/mnt/data/붙여넣은 텍스트(1)(4).txt")
code = path.read_text(encoding="utf-8")

# Add imports for the two comparison models.
code = code.replace(
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.ensemble import RandomForestClassifier\n"
    "from sklearn.linear_model import LogisticRegression\n"
    "from sklearn.tree import DecisionTreeClassifier\n"
)

start_marker = "# --------------------------------------------------\n# 머신러닝: 랜덤 포레스트\n# --------------------------------------------------"
end_marker = "# --------------------------------------------------\n# 새로운 펭귄 종 예측\n# --------------------------------------------------"

start = code.index(start_marker)
end = code.index(end_marker)

replacement = '''# --------------------------------------------------
# 머신러닝: 세 가지 모델 비교
# --------------------------------------------------

st.header("🤖 AI MODEL COMPARISON")
st.subheader("로지스틱 회귀 · 결정 트리 · 랜덤 포레스트 비교")

features = [
    "bill_length_mm",
    "bill_depth_mm",
    "island"
]

X = model_df[features]
y = model_df["species"]

numeric_features = [
    "bill_length_mm",
    "bill_depth_mm"
]

categorical_features = ["island"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# 세 모델에 같은 학습 데이터와 테스트 데이터를 사용합니다.
models = {
    "로지스틱 회귀": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),
    "결정 트리": DecisionTreeClassifier(
        random_state=42
    ),
    "랜덤 포레스트": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1
    )
}

results = []

for model_name, classifier in models.items():
    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )

    model_pipeline.fit(X_train, y_train)
    predictions = model_pipeline.predict(X_test)

    results.append({
        "모델": model_name,
        "정확도": accuracy_score(y_test, predictions),
        "정밀도": precision_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "재현율": recall_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "F1 점수": f1_score(
            y_test, predictions, average="weighted", zero_division=0
        )
    })

results_df = pd.DataFrame(results).set_index("모델")

st.markdown(
    '<div class="section"><h3>📊 MODEL PERFORMANCE</h3>'
    '<p>세 모델을 같은 데이터 분할과 네 가지 평가지표로 비교합니다.</p></div>',
    unsafe_allow_html=True
)

st.dataframe(
    (results_df * 100).round(2).rename_axis("모델"),
    use_container_width=True
)

st.caption("표의 수치는 테스트 데이터로 실제 측정한 결과이며, 실행 환경과 데이터 분할에 따라 달라질 수 있습니다.")

metric_choice = st.selectbox(
    "비교할 평가지표 선택",
    ["정확도", "정밀도", "재현율", "F1 점수"]
)

comparison_chart = px.bar(
    results_df.reset_index(),
    x="모델",
    y=metric_choice,
    color="모델",
    text=results_df[metric_choice].apply(lambda value: f"{value * 100:.2f}%"),
    title=f"모델별 {metric_choice} 비교"
)
comparison_chart.update_layout(
    template="plotly_dark",
    yaxis_tickformat=".0%",
    yaxis_title=metric_choice
)
st.plotly_chart(comparison_chart, use_container_width=True)

# 랜덤 포레스트를 최종 예측 모델로 사용합니다.
random_forest = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1
            )
        )
    ]
)

random_forest.fit(X_train, y_train)

y_pred = random_forest.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

st.subheader("🌲 최종 모델: 랜덤 포레스트")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("정확도", f"{accuracy * 100:.2f}%")

with col2:
    st.metric("정밀도", f"{precision * 100:.2f}%")

with col3:
    st.metric("재현율", f"{recall * 100:.2f}%")

with col4:
    st.metric("F1 점수", f"{f1 * 100:.2f}%")

'''
code = code[:start] + replacement + code[end:]

out = Path("/mnt/data/penguin_beak_model_comparison.py")
out.write_text(code, encoding="utf-8")
print(out, out.stat().st_size)
