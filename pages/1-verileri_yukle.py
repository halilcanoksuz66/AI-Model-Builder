import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np


st.title("📂 Veri Yükleme")
uploaded_file = st.file_uploader("Veri setinizi yükleyin (CSV formatında):")

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        
        # Veri çerçevesinin boş olup olmadığını kontrol et
        if data.empty:
            st.error("Yüklenen dosya boş. Lütfen geçerli bir CSV dosyası yükleyin.")
        else:
            # Veriyi session_state'e kaydet. Javascript Local Storage gibi düşünebiliriz.
            st.session_state['data'] = data
            
            # Yüklenen veri setinin ilk 5 satırını gösterir.
            st.write("Yüklenen Veri:")
            st.write(data.head())


            # Hedef kolonu seçme
            target_column = st.selectbox("Hedef Kolon:", data.columns)
            st.session_state['target_column'] = target_column
            
           
            # Eksik değer kontrolü
            if data.isnull().sum().sum() > 0:
                st.write("🧐 Eksik Değer Kontrolü:")
                st.write(data.isnull().sum())
                missing_values_option = st.radio("Eksik değerleri doldurmak ister misiniz?", ["Hayır", "Ortalama ile doldur", "Medyan ile doldur", "Mod ile doldur"])

                if missing_values_option != "Hayır":
                    try:
                        if missing_values_option == "Ortalama ile doldur":
                            data = data.fillna(data.mean())
                        elif missing_values_option == "Medyan ile doldur":
                            data = data.fillna(data.median())
                        elif missing_values_option == "Mod ile doldur":
                            data = data.fillna(data.mode().iloc[0])

                        # mean() fonksiyonu eksik değerleri doldururken ortalama değeri kullanır.
                        # mode() fonksiyonu eksik değerleri doldururken en sık tekrar eden değeri kullanır. iloc[0] ile ilk değeri alırız.
                        # median() fonksiyonu eksik değerleri doldururken verilerin sıralanmış şekilde ortanca değeri kullanır.
                        # Eksik değerleri doldurulmuş veriyi session_state'e kaydet
                        st.session_state['data'] = data
                        st.success("Eksik değerler başarıyla dolduruldu!")
                        st.write("Eksik Değerleri Doldurulmuş Veri:")
                        st.write(data.head())
                    except Exception as e:
                        st.error(f"Hata oluştu: {e}")


            # Aykırı değerleri tespit ve çıkarma
            def detect_and_remove_outliers(df, threshold=3):
                # StandardScaler kullanarak veriyi standardize et (Z-skoru hesapla)
                # threshold değeri 3 olarak vermemin sebebi verilerin %99.7'sinin -3 ve +3 arasında olmasıdır.
                scaler = StandardScaler()
                z_scores = np.abs(scaler.fit_transform(df))
                
                # Z-skoru > threshold olanları çıkar .all(axis=1) tüm satırları kontrol eder.
                df_cleaned = df[(z_scores < threshold).all(axis=1)]
                
                # Eğer aykırı değerler varsa kullanıcıyı bilgilendir
                if len(df) != len(df_cleaned):
                    st.success(f"Aykırı değerler tespit edildi ve {len(df) - len(df_cleaned)} satır çıkarıldı.")
                else:
                    st.success("Aykırı değer bulunmadı.")

                return df_cleaned

            # Aykırı değerleri temizleme seçeneği
            remove_outliers = st.radio("Aykırı değerleri temizlemek ister misiniz?", ["Hayır", "Evet"])
            if remove_outliers == "Evet":
                try:
                    feature_columns = [col for col in data.columns if col != target_column]
                    print(feature_columns)
                    data_cleaned = detect_and_remove_outliers(data) 

                    # Aykırı değerleri temizlenmiş veriyi session_state'e kaydet
                    st.session_state['data'] = data_cleaned
                    st.write("Aykırı Değerleri Temizlenmiş Veri:")
                    st.write(data_cleaned.head())
                except Exception as e:
                    st.error(f"Hata oluştu: {e}")



            # Veriyi normalleştirme seçeneği
            normalization_method = st.radio("Veriyi normalleştirmek ister misiniz?", ["Hayır", "Min-Max", "Z-Skoru (Standard Scaler)"])

            if normalization_method != "Hayır":
                try:
                    scaler = None
                    if normalization_method == "Min-Max":
                        scaler = MinMaxScaler()
                    elif normalization_method == "Z-Skoru (Standard Scaler)":
                        scaler = StandardScaler()

                    feature_columns = [col for col in data.columns if col != target_column]
                    data_normalized = data.copy()
                    data_normalized[feature_columns] = scaler.fit_transform(data[feature_columns])

                    # Normalleştirilmiş veriyi session_state'e kaydet
                    st.session_state['data'] = data_normalized
                    st.success("Normalizasyon başarıyla uygulandı!")
                    st.write("Normalleştirilmiş Veri:")
                    st.write(data_normalized.head())
                except Exception as e:
                    st.error(f"Hata oluştu: {e}")

             

    except Exception as e:
        st.error(f"Hata oluştu: {e}. Lütfen CSV formatında geçerli bir dosya yükleyin.")
else:
    st.info("Lütfen bir dosya yükleyin.")
