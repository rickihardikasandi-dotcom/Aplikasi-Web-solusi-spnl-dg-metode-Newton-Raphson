import streamlit as st
import numpy as np
import pandas as pd

# Judul Aplikasi
st.set_page_config(page_title="Solver SPNL Newton-Raphson", layout="wide")
st.title("🧮 Solver SPNL - Metode Newton-Raphson")
st.write("Aplikasi untuk mencari solusi sistem persamaan non-linear.")

# Sidebar untuk Input
st.sidebar.header("Konfigurasi Parameter")
tol = st.sidebar.number_input("Toleransi Error", value=1e-6, format="%.e")
max_iter = st.sidebar.number_input("Maksimal Iterasi", value=20)

# Bagian Input Persamaan (Contoh kasus 2 variabel)
st.subheader("1. Definisi Sistem Persamaan")
st.info("Contoh: f1 = x^2 + y^2 - 4 dan f2 = x + y - 2")

col1, col2 = st.columns(2)
with col1:
    x0 = st.number_input("Tebakan awal x:", value=1.0)
with col2:
    y0 = st.number_input("Tebakan awal y:", value=0.5)

# Fungsi Inti Logika
def solve_spnl(x_init, y_init, tolerance, iterations):
    current_x = np.array([x_init, y_init], dtype=float)
    history = []

    for i in range(iterations):
        x, y = current_x[0], current_x[1]
        
        # Definisikan F(x) berdasarkan input (bisa dikembangkan agar dinamis)
        f_val = np.array([
            x**2 + y**2 - 4,
            x + y - 2
        ])
        
        # Definisikan Matriks Jacobian J(x)
        j_val = np.array([
            [2*x, 2*y],
            [1, 1]
        ])
        
        # Hitung Delta (Langkah Newton)
        try:
            delta = np.linalg.solve(j_val, -f_val)
        except np.linalg.LinAlgError:
            st.error("Matriks Jacobian singular! Coba tebakan awal lain.")
            break
            
        current_x = current_x + delta
        error = np.linalg.norm(delta)
        
        history.append({
            "Iterasi": i + 1,
            "x": current_x[0],
            "y": current_x[1],
            "Error": error
        })
        
        if error < tolerance:
            break
            
    return current_x, history

# Tombol Eksekusi
if st.button("Hitung Solusi"):
    hasil, log = solve_spnl(x0, y0, tol, max_iter)
    
    # Tampilkan Hasil Utama
    st.success(f"Solusi ditemukan: x = {hasil[0]:.6f}, y = {hasil[1]:.6f}")
    
    # Tampilkan Tabel Iterasi
    st.subheader("2. Tabel Riwayat Iterasi")
    df_history = pd.DataFrame(log)
    st.table(df_history)
    
    # Grafik Konvergensi (Error)
    st.subheader("3. Grafik Konvergensi Error")
    st.line_chart(df_history.set_index('Iterasi')['Error'])
