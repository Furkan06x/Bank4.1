# -*- coding: utf-8 -*-
"""
Created on 13/12/2024

@author: 

* Furkan Özbek
* Emir Alparslan Dikici
* Berat Yüceldi
* Zeynep Ece Aşkın
"""

import numpy as np
import pickle
import streamlit as st
import matplotlib.pyplot as plt

# Sayfa yapılandırması
st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Model yükleme
try:
    loaded_model = pickle.load(open('bank_model.pkl.sav', 'rb'))
except FileNotFoundError:
    st.error("Model dosyası bulunamadı. Lütfen 'bank_model.pkl.sav' dosyasını kontrol edin.")
    loaded_model = None

# Tahmin fonksiyonu
def prediction_function(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    return prediction

# Ana sayfa fonksiyonu
def main():
    # Metinlerin ayarlanması
    titles = {
        "page_title": "Bank Marketing Prediction Web App",
        "selection": "\U0001F50D Page Selection",
        "tabs": ["Prediction", "Model Performance", "About"],
        "prediction": "\U0001F4CA Prediction",
        "model_performance": "\U0001F3C6 Model Performance",
        "about": "ℹ️ About",
    }

    # Sayfa Seçimi
    st.sidebar.title(titles["selection"])
    page = st.sidebar.radio("Page Select:", titles["tabs"])

    # Tahmin Sayfası
    if page == titles["tabs"][0]:
        st.image("https://i.imgur.com/7rXLpz7.jpeg", use_container_width=True)

        # Sayfa başlığı ve açıklama
        st.markdown("## 📈 **Bank Marketing Prediction Dashboard**", unsafe_allow_html=True)
        st.write("This application predicts whether a customer will subscribe to a bank campaign using data.")

        # Parametre giriş alanları
        st.header("Enter Parameters for Prediction:")
        duration = st.slider(
            "Duration",
            0,
            3700,
            180,
            step=10,
            key="slider_duration"
        )

        previous = st.slider(
            "Previous Contacts",
            0,
            6,
            0,
            step=1,
            key="slider_previous"
        )

        emp_var_rate = st.slider(
            "Employment Variation Rate",
            -3.5,
            1.5,
            0.1,
            step=0.1,
            key="slider_emp_var_rate"
        )

        euribor3m = st.slider(
            "Euribor 3 Month Rate",
            0.6,
            5.1,
            3.6,
            step=0.1,
            key="slider_euribor3m"
        )

        nr_employed = st.slider(
            "Number of Employed",
            4960.0,
            5230.0,
            5166.0,
            step=10.0,
            key="slider_nr_employed"
        )

        col1, col2 = st.columns(2)
        with col1:
            contacted_before = st.radio(
                "Contacted Before",
                ["Yes", "No"],
            )
        with col2:
            contact_cellular = st.radio(
                "Contact Cellular",
                ["Yes", "No"],
            )

        # Tahmin butonu ve sonucu
        if st.button("\U0001F680 Predict"):
            input_data = [
                duration,
                previous,
                emp_var_rate,
                euribor3m,
                nr_employed,
                1 if contacted_before == "Yes" else 0,
                1 if contact_cellular == "Yes" else 0,
            ]
            if loaded_model:
                diagnosis = prediction_function(input_data)

                # Sonuç gösterimi
                if diagnosis == 1:
                    st.success("\u2705 Prediction: Customer will subscribe! (1)")
                else:
                    st.error("\u274C Prediction: Customer will not subscribe. (0)")

    # Model Performansı Sayfası
    elif page == titles["tabs"][1]:
        st.title(titles["model_performance"])
        st.write("Model accuracy and performance metrics are as follows:")
        metrics = {"Accuracy": 0.8931, "Weighted F1 Score": 0.9042, "ROC-AUC Score": 0.9310}
        fig, ax = plt.subplots(figsize=(5, 3))
        bars = ax.bar(metrics.keys(), metrics.values(), color=["#4CAF50", "#FFC107", "#FF5722"])
        ax.set_ylim(0, 1)

        # Add value labels on top of the bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=10
            )

        st.pyplot(fig)

    # Hakkında Sayfası
    elif page == titles["tabs"][2]:
        st.title(titles["about"])
        st.write(
            """
            This application was developed to predict whether a customer will subscribe to a campaign based on banking data.
            The model was trained using the following algorithms for the best performance:
            - Logistic Regression
            - Random Forest Classifier
            - MLP Classifier

            **Developers:**
            - Furkan Özbek  
            - Emir Alparslan Dikici  
            - Berat Yüceldi  
            - Zeynep Ece Aşkın  
            """
        )

# Ana fonksiyon çalıştır
if __name__ == '__main__':
    main()

# Uygulama çalıştırma: streamlit run proj.py
