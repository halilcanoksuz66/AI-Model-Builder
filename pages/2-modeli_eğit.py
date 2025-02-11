import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler  # ROS için gerekli kütüphane
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import StratifiedKFold
# from sklearn.model_selection import train_test_split, cross_val_score

st.title("🛠️ Model Eğitimi")

# Burda local storage ye ekledklerimzi çekiyoruz.
if 'data' in st.session_state and 'target_column' in st.session_state:
    data = st.session_state['data']
    target_column = st.session_state['target_column']

    # Model seçimi
    model_choice = st.selectbox("Model Seçin:", ["Random Forest", "Logistic Regression", "SVM"])
    model = None
    if model_choice == "Random Forest":
        model = RandomForestClassifier()
    elif model_choice == "Logistic Regression":
        model = LogisticRegression()
    else:
        # probalitiy true olayı normalde svm etiket döndür. Ancak böyle yapınca olasılığını döndürttürüyoruz.
        model = SVC(probability=True) 

    handle_imbalance = st.checkbox("Dengesizlikle başa çık (ROS uygula)")


    # Modeli eğit butonuna basıldıysa
    if st.button("Modeli Eğit"):
        data = st.session_state['data']

        # Veri bölme
        # .drop ile istemediklerimizi kaldırırız. Burdada outcome'ı kaldırıyoruz mesela gerisini alıyoruz. 
        X = data.drop(columns=[target_column])
        y = data[target_column]
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # StratifiedKFold nesnesi oluştur
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Skorları ve veri kümelerini saklamak için değişkenler
        best_score = -np.inf # En iyi skorun başlangıç değeri -sonsuz
        best_X_train, best_X_test = None, None
        best_y_train, best_y_test = None, None
        cv_scores = []
        fold = 0
        # Her bir fold için model eğitimi ve değerlendirme
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            fold += 1
            # train_index te tüm ayrılmış satır indexleri var. Yani %80 lik kısım

            # Modeli eğit ve tahmin yap

            if handle_imbalance:
                st.info(f"Fold : {fold} için dengesizlikle başa çıkılıyor... (ROS)")
                ros = RandomOverSampler(random_state=42)  # random state vermemizin sebebi her seferinde aynı sonucu almak. Böyle olunca random olmaz. Modelin tek bir sonucu olur.
                X_train, y_train = ros.fit_resample(X_train, y_train)
                
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Her bir foldun performans analizi için y_test ve y_pred'i saklıyoruz.
            st.session_state[f'y_test_{fold}'] = y_test
            st.session_state[f'y_pred_{fold}'] = y_pred

            # Skoru hesapla
            score = accuracy_score(y_test, y_pred)
            cv_scores.append(score)

            print(f"Fold Skoru: {score:.4f}")

            # En iyi skoru kontrol et ve güncelle
            if score > best_score:
                best_score = score
                best_X_train, best_X_test = X_train, X_test
                best_y_train, best_y_test = y_train, y_test
        # Performans verileri kaydediliyor
                st.session_state['model'] = model
                st.session_state['y_test'] = best_y_test
                st.session_state['y_pred'] = y_pred
                st.session_state['cv_scores'] = cv_scores  

        st.success("Model eğitildi ve metrikler hesaplandı!")
        st.write("📊 K-Fold CV Skorları:")
        st.write(cv_scores)


else:
    st.warning("Lütfen önce veri yükleyin ve hedef kolonu seçin!")



  # x train : outcome hariicindeki o %80 lik seçilen sütun
        # x test : outcome haricindeki o ayrılan % 20 lu kısım 
        # y train : outcome inin %80 lik kısmı yani üstteki xtrainin sonuçları gibi düşünebilirz.
        # y test :  outcome nin %20 luk kısmı bh da üstteki xtesttin sonuçkarı gibi düşünebiliriz.

  # cv_scores = cross_val_score(model, X_train, y_train, cv=5)
         # Burda normalde tüm veri setini vermemiz gerekir K folda. Benim verememim sebebi ise amacım gizli bir validation set oluşturmak. bunun skorunu alıyoruz. Böylelikle test seti üzerinde doğru bir tahmin yapmış oluyoruz. Yani overfittingten kaçıyorum. Ancak bu durumda underfitting yaşasaydım yani skorum düşük çıksaydı bu durumda tüm veri setini verip modeli eğitmek daha mantıklı olurdu.

        # Model eğitimi ve tahmin
        # model.fit(best_X_train, best_y_train)
        # best_y_pred = model.predict(best_X_test)
