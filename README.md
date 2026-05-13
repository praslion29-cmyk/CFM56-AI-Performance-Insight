# CFM56-AI-Performance-Insight
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import base64

# ===================== CONFIG =====================
st.set_page_config(page_title="CFM56 Engine Monitor", layout="wide")

# ===================== HEADER WITH LOGO =====================
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

img_base64 = get_base64_image("cfm56.png")

if img_base64:
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:15px;">
        <img src="data:image/png;base64,{img_base64}" width="80">
        <div>
            <h1 style="margin:0;">CFM56-7B Engine Health Monitor</h1>
            <p style="margin:0; color:gray;">Aircraft Engine Performance Analytics</p>
        </div>
    </div>
    <hr>
    """, unsafe_allow_html=True)
else:
    st.title("CFM56-7B Engine Health Monitor")

# ===================== NARRATIVE =====================
st.markdown("""
### ✈️ About This System
Platform ini digunakan untuk menganalisis kesehatan engine berbasis data dan machine learning.

Fitur utama:
- Deteksi kesehatan engine
- Analisis tren parameter
- Deteksi anomali
- Rekomendasi maintenance

Upload data CSV/XLSX untuk mulai analisis.
""")

# ===================== INPUT =====================
col1, col2 = st.columns(2)

with col1:
    reg = st.text_input("Aircraft Registration", "PK-XXX")
with col2:
    eng = st.selectbox("Engine Position", ["ENG 1", "ENG 2"])

file = st.file_uploader("Upload Data", type=["csv", "xlsx"])

# ===================== HEALTH BAR =====================
def show_health_bar(score):
    st.markdown(f"""
    <div style="margin-top:20px;">
        <div style="width:100%;background: linear-gradient(to right, red, yellow, green);height: 20px;border-radius:10px;position:relative;">
            <div style="position:absolute;left:{score}%;top:-5px;transform:translateX(-50%);">⬇️</div>
        </div>
        <p style="text-align:center;">Health Score: {score:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)

# ===================== MAIN =====================
if file:
    df = pd.read_csv(file) if file.name.endswith("csv") else pd.read_excel(file)

    st.subheader("Preview Data")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=['float64','int64']).columns
    X = df[numeric_cols].dropna()

    if len(X) > 10:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        preds = model.fit_predict(X_scaled)

        X["anomaly"] = preds
        df.loc[X.index, "anomaly"] = preds
        df["anomaly_label"] = df["anomaly"].map({1:"Normal", -1:"Anomaly"})

        anomalies = X[X["anomaly"] == -1]
        anomaly_ratio = len(anomalies) / len(X)
        health_score = max(0, 100 - anomaly_ratio*100)

        status = "🟢 HEALTHY" if health_score>80 else "🟡 WARNING" if health_score>50 else "🔴 CRITICAL"

        # KPI
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Health Score", f"{health_score:.1f}%")
        c2.metric("Total Data", len(df))
        c3.metric("Anomaly", len(anomalies))
        c4.metric("Anomaly %", f"{anomaly_ratio*100:.2f}%")

        st.subheader(f"Engine Status: {status}")
        show_health_bar(health_score)

        # TIME DETECTION
        time_col=None
        for col in df.columns:
            if 'time' in col.lower() or 'date' in col.lower():
                time_col=col
                break

        if time_col:
            df[time_col]=pd.to_datetime(df[time_col], errors='coerce')

        # SINGLE GRAPH
        st.subheader("Parameter Trends")
        param=st.selectbox("Pilih Parameter", numeric_cols)

        x_axis=df[time_col] if time_col else df.index

        fig, ax = plt.subplots()
        ax.plot(x_axis, df[param])
        ax.set_title(param)
        ax.set_xlabel("Time" if time_col else "Index")
        ax.set_ylabel(param)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # ===================== PCA VISUALIZATION =====================
        from sklearn.decomposition import PCA

        st.subheader("PCA Visualization (2D Projection)")

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
        pca_df["anomaly_label"] = df.loc[X.index, "anomaly_label"].values

        fig_pca = px.scatter(
            pca_df,
            x="PC1",
            y="PC2",
            color="anomaly_label",
            title="PCA Projection of Engine Data"
        )

        st.plotly_chart(fig_pca, use_container_width=True)

        # PLOTLY ANOMALY
        st.subheader("Anomaly Detection (Interactive)")
        anomaly_param = st.selectbox("Pilih Parameter (Anomaly)", numeric_cols, key="plotly")

        fig2 = px.scatter(
            df.loc[X.index],
            x=time_col if time_col else df.loc[X.index].index,
            y=anomaly_param,
            color="anomaly_label",
            hover_data=numeric_cols,
            title=f"Anomaly Detection: {anomaly_param}"
        )

        st.plotly_chart(fig2, use_container_width=True)

        # TABLE
        st.subheader("Detected Anomalies")
        st.dataframe(anomalies.head(20))

        # RECOMMENDATION
        st.subheader("AI Recommendation")
        if health_score>80:
            st.success("Engine normal")
        elif health_score>50:
            st.warning("Perlu pengecekan")
        else:
            st.error("Perlu maintenance segera")

    else:
        st.warning("Data kurang")
