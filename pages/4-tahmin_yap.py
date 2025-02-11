import streamlit as st
import numpy as np

st.title("🔮 Tahmin Yapma")
# Local Storage'deki verileri çektik.
if 'model' in st.session_state and 'data' in st.session_state:
    model = st.session_state['model']
    data = st.session_state['data']

    # Eğitimde kullanılan özellikler : outcome hariç hepsini alıp yazdırıyoruz listeyi. 
    feature_columns = [col for col in data.columns if col != st.session_state['target_column']]
    st.write("Modelin Beklediği Özellikler:")
    st.write(feature_columns)

    # Kullanıcıdan giriş al ve user_input listesine ekle
    user_input = []
    for feature in feature_columns:
        value = st.text_input(f"{feature} için değer girin:", "")
        user_input.append(value)

    # Tahmin yap butonuna basıldıysa
    if st.button("Tahmin Yap"):
        try:
            # Girdileri numpy array'e çevir. reshape(1, -1) ile 2D array'e dönüştür. 
            # Bunu yapmamızın sebebi modelin 2D array beklemesi.
            input_array = np.array(user_input, dtype=float).reshape(1, -1)

            # Tahmin yap ve sonuçları yazdır
            # predict ile predict_proba arasındaki fark predict_proba olasılık döndürürken predict etiket döndürür.
            prediction = model.predict(input_array)
            prediction_proba = model.predict_proba(input_array)

            # Yalnızca bir tahmin dizisi gönderdiğimiz için ilk elemanı alıyoruz.
            st.write("📌 Tahmin Sonucu:")
            st.write(f"- Sınıf: {prediction[0]}")
            st.write(f"- Olasılıklar: {prediction_proba[0]}")
        except ValueError as e:
            st.error(f"Girdiğiniz değerlerde hata var: {e}")
else:
    st.warning("Lütfen önce modeli eğitin ve veriyi yükleyin!")
