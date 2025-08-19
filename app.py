import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# Judul
# =========================
st.set_page_config(page_title="Prediksi Risiko Kredit Mekaar PNM", layout="wide")
st.title("🔍 Deteksi Dini Risiko Kredit Mekaar PNM")
st.markdown("Aplikasi interaktif untuk analisis data pinjaman dan prediksi risiko kredit nasabah berbasis machine learning.")

# =========================
# Upload Dataset
# =========================
st.sidebar.header("📂 Upload Data Nasabah")
uploaded_file = st.sidebar.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📋 Data Nasabah (Preview)")
    st.dataframe(df.head())

    # =========================
    # Preprocessing
    # =========================
    if "NoOfArrearDays" in df.columns:
        def label_risk(days):
            if days == 0:
                return 0  # rendah
            elif 1 <= days <= 30:
                return 1  # sedang
            else:
                return 2  # tinggi

        df["risk_level"] = df["NoOfArrearDays"].apply(label_risk)

        # Fitur numerik saja
        features = ["ODInterest", "ODPrincipal", "PrincipalDue", "InterestDue", "NoOfArrearDays"]
        df = df.dropna(subset=features)

        X = df[features]
        y = df["risk_level"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        # =========================
        # Training Model
        # =========================
        model = RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # =========================
        # Evaluasi
        # =========================
        st.subheader("📊 Evaluasi Model")
        st.text(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        im = ax.imshow(cm, cmap="Blues")

        ax.set_xticks(np.arange(3))
        ax.set_yticks(np.arange(3))
        ax.set_xticklabels(["Rendah", "Sedang", "Tinggi"])
        ax.set_yticklabels(["Rendah", "Sedang", "Tinggi"])
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="red")

        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

        # =========================
        # Input Prediksi Manual
        # =========================
        st.sidebar.subheader("🔮 Prediksi Risiko Baru")

        ODInterest = st.sidebar.number_input("ODInterest", min_value=0.0)
        ODPrincipal = st.sidebar.number_input("ODPrincipal", min_value=0.0)
        PrincipalDue = st.sidebar.number_input("PrincipalDue", min_value=0.0)
        InterestDue = st.sidebar.number_input("InterestDue", min_value=0.0)
        NoOfArrearDays = st.sidebar.number_input("NoOfArrearDays", min_value=0)

        if st.sidebar.button("Prediksi Risiko"):
            input_data = pd.DataFrame([[
                ODInterest, ODPrincipal, PrincipalDue, InterestDue, NoOfArrearDays
            ]], columns=features)

            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            proba = model.predict_proba(input_scaled)[0]

            st.subheader("📌 Hasil Prediksi")
            if prediction == 0:
                st.success(f"✅ Risiko Rendah — Probabilitas: {proba[prediction]:.2%}")
            elif prediction == 1:
                st.warning(f"⚠ Risiko Sedang — Probabilitas: {proba[prediction]:.2%}")
            else:
                st.error(f"🚨 Risiko Tinggi — Probabilitas: {proba[prediction]:.2%}")
