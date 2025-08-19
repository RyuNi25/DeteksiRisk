import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

st.set_page_config(page_title="Deteksi Risiko Kredit PNM", layout="wide")

st.title("🔍 Deteksi Dini Risiko Kredit Mekaar PNM")
st.markdown("Upload dataset pinjaman, sistem akan otomatis analisis & training model prediksi risiko kredit.")

# =======================
# Upload Dataset
# =======================
uploaded_csv = st.file_uploader("📂 Upload file CSV data nasabah", type=["csv"])

if uploaded_csv is not None:
    df = pd.read_csv(uploaded_csv)

    st.subheader("📊 Ringkasan Data")
    st.write(df.head())
    st.write(df.describe())

    # =======================
    # Preprocessing
    # =======================
    # Misalnya: target = "Label" (0 = risiko rendah, 1 = risiko tinggi)
    target_col = "Label"  # ganti sesuai datasetmu
    if target_col in df.columns:
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # =======================
        # Training Model
        # =======================
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Evaluasi
        y_pred = model.predict(X_test)
        st.subheader("📈 Evaluasi Model")
        st.text(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Rendah","Tinggi"], yticklabels=["Rendah","Tinggi"])
        st.pyplot(fig)

        # =======================
        # Form Prediksi
        # =======================
        st.subheader("📝 Prediksi Risiko Kredit Baru")
        input_data = {}
        for col in X.columns:
            val = st.number_input(f"{col}", value=float(df[col].mean()))
            input_data[col] = val

        if st.button("Prediksi"):
            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)
            pred = model.predict(input_scaled)[0]
            proba = model.predict_proba(input_scaled)[0][1]

            if pred == 1:
                st.error(f"⚠ Risiko Tinggi — Probabilitas: {proba:.2%}")
            else:
                st.success(f"✅ Risiko Rendah — Probabilitas: {proba:.2%}")

        # =======================
        # Feature Importance
        # =======================
        st.subheader("📌 Faktor Terpenting dalam Prediksi")
        feature_importance = pd.DataFrame({
            "Fitur": X.columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        st.bar_chart(feature_importance.set_index("Fitur"))
    else:
        st.warning("⚠ Kolom target 'Label' tidak ditemukan di dataset. Pastikan dataset punya kolom target untuk training.")
