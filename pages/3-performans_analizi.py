import streamlit as st
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

st.title("📊 Performans Analizi")

# Kontrol: Gerekli tüm öğeler session_state'de mevcut mu?
required_keys = [
    'model', 'y_test', 'y_pred', 'cv_scores', 
    'y_test_1', 'y_pred_1', 'y_test_2', 'y_pred_2', 
    'y_test_3', 'y_pred_3', 'y_test_4', 'y_pred_4', 
    'y_test_5', 'y_pred_5'
]

if all(key in st.session_state for key in required_keys):
    # Session state'teki değerleri alın
    fold_choice = st.selectbox("Fold Seçin:", ["en iyi", "1", "2", "3", "4", "5"])
    
    # Fold'a göre test ve tahmin değerlerini ayarla
    if fold_choice != "en iyi":
        y_test = st.session_state[f'y_test_{fold_choice}']
        y_pred = st.session_state[f'y_pred_{fold_choice}']
    else:
        y_test = st.session_state['y_test']
        y_pred = st.session_state['y_pred']

    # Karışıklık Matrisi
    st.subheader("🌀 Karışıklık Matrisi:")
    cm = confusion_matrix(y_test, y_pred)

    # Matris Görselleştirme
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=True, yticklabels=True, ax=ax)
    # annot=True sayesinde her bir kareye değerini yazar. fmt='d' sayesinde sayısal değerler yazılır. cmap ile renk belirleriz.
    # xticklabels ve yticklabels ile etiketleri gösteririz. ax=ax ile grafiği çizdiririz.
    ax.set_xlabel("Tahmin Edilen")
    ax.set_ylabel("Gerçek")
    st.pyplot(fig) # Matrisi gösterir.

    # Performans Metrikleri
    st.subheader("📈 Performans Metrikleri:")
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    st.write(f"- **Doğruluk (Accuracy):** {accuracy:.2f}")
    st.write(f"- **Duyarlılık (Recall):** {recall:.2f}")
    st.write(f"- **Kesinlik (Precision):** {precision:.2f}")
    st.write(f"- **F1 Skoru:** {f1:.2f}")

    # K-Fold CV Skorları
    st.subheader("📊 K-Fold CV Skorları:")
    st.write(st.session_state['cv_scores'])
else:
    st.warning("Lütfen önce modeli eğitin ve gerekli verileri yükleyin!")
