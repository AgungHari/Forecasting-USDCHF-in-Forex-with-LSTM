from keras.models import load_model
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Fungsi untuk memuat model dan melakukan prediksi
def predict_stock_price(model_path, new_data):

    scaler = joblib.load('scaler_ta_abel1.save')

    # Memuat model yang sudah disimpan
    model = load_model(model_path)
    print("Model berhasil dimuat dari", model_path)
    
    # Data baru yang ingin diprediksi, harus berupa numpy array
    # Misalkan data yang kamu masukkan dalam format list
    data_new = np.array(new_data).reshape(-1, 1)
    
    # Normalisasi data baru (fit di sini hanya contoh, seharusnya disesuaikan dengan scaler yang sama saat training)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_new_scaled = scaler.transform(data_new)
    
    # Membentuk input yang sesuai untuk model LSTM
    X_input = data_new_scaled.reshape(1, data_new_scaled.shape[0], 1)
    
    # Melakukan prediksi
    predicted_price_scaled = model.predict(X_input)
    
    # Mengembalikan prediksi ke skala harga asli
    predicted_price = scaler.inverse_transform(predicted_price_scaled)
    
    print(f"Prediksi harga saham: {predicted_price[0][0]}")
    return predicted_price[0][0]

# Contoh penggunaan fungsi
new_data = [0.98009, 0.98100, 0.98096, 0.98111, 0.98222]  # Data input baru untuk prediksi
model_path = 'model_ta_abel1.h5'  # Path model yang disimpan

predicted_price = predict_stock_price(model_path, new_data)
print("Prediksi harga saham untuk data baru:", predicted_price)
