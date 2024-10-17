from keras.models import load_model
import numpy as np
import joblib  # untuk memuat scaler yang sudah disimpan
import matplotlib.pyplot as plt

# Fungsi untuk memuat model, melakukan prediksi, dan menampilkan grafik
async def predict_and_plot_stock_price(model_path, new_data):
    # Memuat scaler yang sudah di-fit saat training
    scaler = joblib.load('scaler_ta_abel1.save')

    # Memuat model yang sudah disimpan
    model = load_model(model_path)
    print("Model berhasil dimuat dari", model_path)
    
    # Data baru yang ingin diprediksi, harus berupa numpy array
    data_new = np.array(new_data).reshape(-1, 1)  # Sesuaikan format data baru
    
    # Normalisasi data baru menggunakan scaler yang sudah di-fit pada training
    data_new_scaled = scaler.transform(data_new)
    
    # Membentuk input yang sesuai untuk model LSTM (1 sample, 20 timesteps, 1 feature)
    X_input = data_new_scaled.reshape(1, data_new_scaled.shape[0], 1)
    
    # Melakukan prediksi (output adalah prediksi untuk 5 timestep ke depan)
    predicted_price_scaled = model.predict(X_input)
    
    # Mengembalikan prediksi ke skala harga asli (5 prediksi)
    predicted_price = scaler.inverse_transform(predicted_price_scaled)
    predicted_price = predicted_price.flatten()  # Mengubah ke bentuk 1D array
    
    # Menampilkan prediksi
    print(f"Prediksi harga saham (5 timestep ke depan): {predicted_price}")
    
    # Gabungkan data historis dengan prediksi untuk plotting
    combined_data = np.concatenate((new_data, predicted_price))
    
    # Plot keseluruhan data dengan warna biru terlebih dahulu
    plt.plot(np.arange(1, len(combined_data)+1), combined_data, label='Data Input & Prediksi', color='blue')
    
    # Plot bagian prediksi dengan warna merah (dimulai dari titik prediksi)
    plt.plot(np.arange(len(new_data)+1, len(combined_data)+1), combined_data[len(new_data):], color='red')
    
    # Menambahkan label dan judul
    plt.title('Grafik Data Input dan Prediksi 5 Timestep ke Depan')
    plt.xlabel('Timestep')
    plt.ylabel('Harga Saham')
    plt.legend()
    
    # Menampilkan plot
    plt.grid(True)
    plt.show()
    
    return predicted_price

# Contoh penggunaan fungsi
new_data = [0.85192, 0.85256, 0.85242, 0.85234, 0.85232, 0.8523, 0.85229, 0.85228, 0.85229, 0.85227, 
            0.85228, 0.85217, 0.85212, 0.85208, 0.85207, 0.85211, 0.85206, 0.85197, 0.85193, 0.85194]
 # Data input 20 timesteps

model_path = 'model_ta_abel1.h5'  # Path model yang disimpan

await predict_and_plot_stock_price(model_path, new_data)
