import streamlit as st
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from datetime import datetime
import plotly.express as px
import pandas as pd

# ======================
# Konfigurasi Halaman Utama
# ======================
st.set_page_config(
    page_title="IkanCheck",
    page_icon="🐟",
    layout="wide"
)

# ======================
# CSS Kustom untuk Tampilan
# ======================
st.markdown("""
    <style>
    /* CSS untuk layout full-width */
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100% !important;
    }
    /* CSS untuk Card di Halaman Beranda */
    .card {
        background-color: #262730;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        border: 1px solid #4D4D4D;
    }
    .card h2, .card h3 {
        color: #FFFFFF;
        margin-top: 0;
    }
    </style>
""", unsafe_allow_html=True)

# ======================
# Pemuatan Model & Konfigurasi Awal
# ======================
# Gunakan cache untuk memuat model agar tidak diulang setiap kali ada interaksi
@st.cache_resource
def load_keras_model():
    model = load_model("model999.h5")
    return model

model = load_keras_model()

class_labels = {
    "Bacterial Red disease": 0,
    "Bacterial diseases - Aeromoniasis": 1,
    "Bacterial gill disease": 2,
    "Fungal diseases Saprolegniasis": 3,
    "Healthy Fish": 4,
    "Parasitic diseases": 5,
    "Viral diseases White tail disease": 6,
    "bukan ikan": 7
}
idx_to_class = {v: k for k, v in class_labels.items()}

HISTORY_DIR = "riwayat_upload"
os.makedirs(HISTORY_DIR, exist_ok=True)

# ======================
# Database Teks (Saran & Edukasi)
# ======================
saran_pengobatan = {
    "Bacterial Red disease": """
        **Penyebab:** Bakteri *Aeromonas hydrophila* atau *Pseudomonas sp.*, sering dipicu oleh stres atau kualitas air yang buruk.
        **Gejala Umum:** Bercak merah atau borok pada tubuh, sirip, atau ekor. Ikan menjadi lesu dan kehilangan nafsu makan.
        **Saran Pengobatan:**
        1.  **Karantina:** Segera pisahkan ikan yang sakit untuk mencegah penularan.
        2.  **Perbaikan Kualitas Air:** Lakukan penggantian air sekitar 30-50% dan pastikan parameter air (pH, amonia) stabil.
        3.  **Pengobatan:** Gunakan antibiotik yang sesuai seperti Oxytetracycline atau Enrofloxacin yang dicampurkan ke dalam pakan atau air sesuai dosis yang dianjurkan.
        4.  **Garam Ikan:** Tambahkan garam ikan (non-yodium) sekitar 1-3 gram per liter air untuk membantu mengurangi stres osmotik pada ikan.
    """,
    "Bacterial diseases - Aeromoniasis": """
        **Penyebab:** Infeksi bakteri *Aeromonas*, biasanya menyerang ikan yang sedang stres atau terluka.
        **Gejala Umum:** Perut kembung (dropsy), mata menonjol (pop-eye), borok pada kulit.
        **Saran Pengobatan:**
        1.  Segera karantina ikan yang terinfeksi.
        2.  Gunakan antibiotik seperti Kanamycin atau Metronidazole. Konsultasikan dosis dengan ahli.
        3.  Jaga kebersihan akuarium secara maksimal.
    """,
    "Bacterial gill disease": """
        **Penyebab:** Bakteri seperti *Flavobacterium branchiophilum*. Sering terjadi di akuarium padat dengan kualitas air rendah.
        **Gejala Umum:** Insang terlihat bengkak, pucat, atau tertutup lendir. Ikan kesulitan bernapas dan sering megap-megap di permukaan.
        **Saran Pengobatan:**
        1.  Tingkatkan aerasi dan kualitas air.
        2.  Lakukan perendaman dengan larutan garam ikan.
        3.  Gunakan pengobatan antibakteri seperti Acriflavine atau Formalin sesuai petunjuk.
    """,
    "Fungal diseases Saprolegniasis": """
        **Penyebab:** Jamur *Saprolegnia sp.*, biasanya menyerang jaringan tubuh ikan yang sudah ada luka sebelumnya.
        **Gejala Umum:** Tumbuh lapisan seperti kapas berwarna putih atau keabu-abuan pada kulit, sirip, atau mata ikan.
        **Saran Pengobatan:**
        1.  Gunakan anti-jamur seperti Malachite Green atau Methylene Blue.
        2.  Jaga suhu air tetap stabil dan bersih.
        3.  Pindahkan ikan ke tank karantina selama pengobatan.
    """,
    "Parasitic diseases": """
        **Penyebab:** Parasit eksternal seperti *Ichthyophthirius multifiliis* (White Spot) atau kutu ikan (*Argulus*).
        **Gejala Umum:** Bintik-bintik putih di seluruh tubuh (White Spot), ikan sering menggesekkan tubuhnya ke benda, atau terlihat parasit menempel di kulit.
        **Saran Pengobatan:**
        1.  Naikkan suhu air secara bertahap (untuk White Spot) ke 28-30°C untuk mempercepat siklus hidup parasit.
        2.  Gunakan obat anti-parasit yang mengandung Malachite Green dan Formalin.
        3.  Jaga kebersihan dasar akuarium karena beberapa parasit berkembang biak di substrat.
    """,
    "Viral diseases White tail disease": """
        **Penyebab:** Infeksi virus. Penyakit ini sangat menular dan seringkali sulit diobati.
        **Gejala Umum:** Muncul warna putih buram atau seperti susu pada bagian pangkal hingga ujung ekor.
        **Saran Pengobatan:**
        1.  Saat ini belum ada obat antivirus yang efektif untuk ikan.
        2.  Fokus utama adalah **pencegahan**: jaga kualitas air, berikan nutrisi terbaik untuk meningkatkan imun ikan.
        3.  Segera pisahkan ikan yang terinfeksi untuk mencegah wabah.
    """,
    "Healthy Fish": "Ikan Anda terlihat sehat! Terus jaga kualitas air dan berikan pakan yang baik."
}

edukasi_lengkap = {
    "Bacterial Red disease": {
        "img": "image/bacterial.jpg",
        "nama_lain": "Penyakit Bercak Merah, *Epizootic Ulcerative Syndrome (EUS)*",
        "penyebab": "Umumnya disebabkan oleh infeksi bakteri seperti *Aeromonas hydrophila* atau *Pseudomonas sp.*. Seringkali dipicu oleh stres, kualitas air yang buruk, atau luka pada tubuh ikan.",
        "gejala_umum": """
        - Munculnya bercak merah, ruam, atau luka borok (ulkus) pada kulit, sirip, atau pangkal ekor.
        - Sirip terlihat geripis atau rusak.
        - Ikan menjadi lesu, kehilangan nafsu makan, dan sering menyendiri.
        - Pada kasus parah, dapat terjadi pendarahan di beberapa bagian tubuh.
        """,
        "penanganan_dan_pengobatan": """
        1.  **Karantina**: Segera pisahkan ikan yang sakit ke dalam tangki karantina untuk mencegah penularan.
        2.  **Perbaikan Kualitas Air**: Lakukan penggantian air parsial (30-50%) secara rutin. Pastikan parameter air seperti pH, amonia, dan nitrit berada pada level yang aman.
        3.  **Penggunaan Antibiotik**: Lakukan perendaman atau campurkan antibiotik seperti *Oxytetracycline* atau *Enrofloxacin* ke dalam pakan. Selalu ikuti dosis dan petunjuk penggunaan.
        4.  **Pemberian Garam Ikan**: Tambahkan garam ikan (non-yodium) dengan dosis 1-3 gram per liter air di tangki karantina untuk membantu proses penyembuhan dan mengurangi stres osmotik.
        """,
        "pencegahan": """
        - Jaga kebersihan akuarium dan filter secara rutin.
        - Hindari kepadatan populasi ikan yang berlebihan.
        - Berikan pakan yang berkualitas dan bervariasi.
        """
    },
    # Note: Ensure you have complete entries for all other diseases here.
    "Healthy Fish": {
        "img": "image/healty.jpg",
        "nama_lain": "Ikan Sehat",
        "penyebab": "Kondisi ideal yang dicapai melalui perawatan yang baik.",
        "gejala_umum": """
        - Berenang aktif dan responsif.
        - Warna tubuh cerah, cerah, dan tidak kusam.
        - Sirip dan ekor mengembang sempurna dan tidak ada sobekan.
        - Mata jernih, tidak berkabut atau menonjol.
        - Tidak ada bintik, bercak, luka, atau lapisan lendir aneh pada tubuh.
        - Bernapas dengan tenang dan memiliki nafsu makan yang baik.
        """,
        "penanganan_dan_pengobatan": "Tidak diperlukan pengobatan. Teruskan perawatan yang baik.",
        "pencegahan": "Kualitas air, pakan bergizi, dan lingkungan bebas stres adalah tiga pilar utama untuk menjaga ikan tetap sehat."
    },
    # ... (and so on for all other diseases)
}


# ======================
# Fungsi Prediksi
# ======================
def model_prediction(img):
    img = img.resize((299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0
    preds = model.predict(x)
    # Mengembalikan seluruh array probabilitas prediksi
    return preds[0]

# ======================
# Sidebar Navigasi
# ======================
st.sidebar.title("🧭 Navigasi")
page = st.sidebar.selectbox("Pilih Halaman", ["🏠 Beranda", "🔍 Deteksi Penyakit", "📚 Edukasi Penyakit", "📝 Riwayat", "ℹ️ Tentang"])


# ======================
# ----- HALAMAN BERANDA -----
# ======================
if page == "🏠 Beranda":
    st.title("🐟 Selamat Datang di IkanCheck")
    st.subheader("Sistem Deteksi Penyakit Ikan Air Tawar Berbasis CNN Xception")
    st.markdown("---")

    # --- Card 1: Apa itu IkanCheck? ---
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    with col1:
        try:
            img = Image.open("bgikan.jpg")
            st.image(img, caption="IkanCheck", use_container_width=True)
        except FileNotFoundError:
            st.error("Gambar bgikan.jpg tidak ditemukan!")
    with col2:
        st.markdown("""
            <h2>🎯 Apa itu IkanCheck?</h2>
            <p>IkanCheck adalah aplikasi berbasis <b>AI (Convolutional Neural Network - Xception)</b> 
            yang dapat mendeteksi penyakit pada ikan air tawar hanya dengan menggunakan <b>gambar</b>.</p>
            <ul>
                <li>🔹 Mudah digunakan</li>
                <li>🔹 Cepat & akurat</li>
                <li>🔹 Edukasi lengkap penyakit ikan</li>
            </ul>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Card 2: Statistik Singkat ---
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h3>📊 Statistik Singkat</h3>", unsafe_allow_html=True)
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    stat_col1.metric("Jenis Penyakit", "6+1")
    stat_col2.metric("Algoritma", "CNN Xception")
    stat_col3.metric("Dataset", "Kaggle")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Card 3: Tips Cepat ---
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h3>💡 Tips Cepat</h3>", unsafe_allow_html=True)
    st.markdown("""
        <ol>
            <li>Jaga kualitas air tetap bersih</li>
            <li>Berikan pakan bergizi</li>
            <li>Amati perubahan perilaku ikan</li>
        </ol>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br><center>✨ Dibuat dengan ❤️ menggunakan Streamlit ✨</center>", unsafe_allow_html=True)


# ===================================================================
# ----- HALAMAN DETEKSI (VERSI BARU DENGAN GRAFIK) -----
# ===================================================================
elif page == "🔍 Deteksi Penyakit":
    st.title("🔍 Deteksi Penyakit Ikan")
    st.info("Unggah gambar ikan Anda untuk memulai deteksi. Untuk hasil terbaik, ikuti tips di samping.")

    col1, col2 = st.columns([2, 1])

    with col2:
        st.subheader("💡 Tips Foto Akurat")
        st.markdown("""
        - **Gunakan Cahaya Cukup:** Pastikan ikan terlihat jelas.
        - **Fokus Jelas:** Hindari gambar yang buram.
        - **Tampilkan Gejala:** Arahkan kamera pada bagian tubuh ikan yang aneh.
        - **Satu Ikan per Foto:** Fokus pada satu ikan untuk hasil terbaik.
        """)

    with col1:
        uploaded_file = st.file_uploader("Pilih atau seret gambar ikan ke sini", 
                                         type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB')
        with col1:
            st.image(img, caption="Gambar yang akan dideteksi", width=500) 
            
            if st.button("Deteksi Sekarang"):
                with st.spinner('Menganalisis gambar...'):
                    # Dapatkan semua probabilitas prediksi
                    all_predictions = model_prediction(img)
                    
                    # Dapatkan kelas dan keyakinan tertinggi
                    pred_class_idx = np.argmax(all_predictions)
                    confidence = np.max(all_predictions)
                    label = idx_to_class[pred_class_idx]

                    AMBANG_BATAS = 0.70

                    # 1. Cek PERTAMA: Apakah hasilnya adalah "bukan ikan"?
                    if label == "bukan ikan":
                        st.divider()
                        st.error("❌ Gambar Tidak Valid")
                        st.warning("Gambar yang Anda unggah tidak terdeteksi sebagai ikan. Mohon unggah foto ikan yang jelas sesuai dengan tips.")

                    # 2. Cek KEDUA: Jika bukan "bukan ikan", apakah keyakinannya terlalu rendah?
                    elif confidence < AMBANG_BATAS:
                        st.divider()
                        st.warning(f"⚠️ Model Ragu")
                        st.info(f"Model hanya memiliki keyakinan sebesar {confidence*100:.2f}%. Ini terlalu rendah untuk memberikan hasil yang akurat.")
                        st.markdown("Pastikan gambar yang diunggah adalah foto ikan air tawar yang jelas.")

                    # 3. KETIGA: Jika lolos dua cek di atas, baru tampilkan hasil deteksi penyakit
                    else:
                        # --- Tampilkan hasil di kolom 1 ---
                        st.divider()
                        st.success(f"Hasil Deteksi: **{label}**")
                        st.info(f"Tingkat Keyakinan: {confidence*100:.2f}%")
                        
                        saran = saran_pengobatan.get(label, "Tidak ada saran spesifik.")
                        with st.expander("🔬 **Lihat Detail dan Saran Penanganan**"):
                            st.markdown(saran)

                        # --- Simpan riwayat ---
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        save_path = os.path.join(HISTORY_DIR, f"{timestamp}_{label}.jpg")
                        img.save(save_path)
                        
                        # --- Buat dan Tampilkan Grafik di kolom 2 ---
                        # Buat DataFrame untuk grafik
                        df = pd.DataFrame({
                            'Penyakit': list(idx_to_class.values()),
                            'Keyakinan': all_predictions * 100
                        })
                        df = df.sort_values(by='Keyakinan', ascending=True)

                        # Buat grafik bar horizontal dengan Plotly
                        fig = px.bar(
                            df,
                            x='Keyakinan',
                            y='Penyakit',
                            orientation='h',
                            title='Grafik Keyakinan Model',
                            labels={'Keyakinan': 'Keyakinan (%)'},
                            text=df['Keyakinan'].apply(lambda x: f'{x:.2f}%')
                        )
                        fig.update_layout(
                            template='plotly_dark',
                            xaxis_title="Keyakinan (%)",
                            yaxis_title="",
                            height=350
                        )
                        fig.update_traces(marker_color='#33FF8A') # Warna hijau agar serasi

                        # Tampilkan di bawah tips
                        col2.subheader("Distribusi Keyakinan")
                        col2.plotly_chart(fig, use_container_width=True)

    else:
        # Jika tidak ada file yang diunggah, tampilkan contoh (opsional)
        with col1:
            st.divider()
            st.write("Belum ada gambar yang diunggah.")

# ======================
# ----- HALAMAN EDUKASI -----
# ======================
elif page == "📚 Edukasi Penyakit":
    st.title("📚 Penjelasan Penyakit Ikan")
    st.markdown("Pelajari lebih lanjut tentang berbagai kondisi yang dapat mempengaruhi ikan air tawar.")

    nama_penyakit_list = list(edukasi_lengkap.keys())
    
    if nama_penyakit_list:
        selected_disease = st.selectbox("Pilih Penyakit untuk Dilihat Detailnya", nama_penyakit_list)
        
        konten = edukasi_lengkap[selected_disease]
        
        st.header(selected_disease)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            try:
                st.image(konten["img"], use_container_width=True)
            except (FileNotFoundError, KeyError):
                st.warning(f"Gambar untuk {selected_disease} tidak ditemukan.")

        with col2:
            if "nama_lain" in konten:
                st.markdown(f"**Nama Lain:** *{konten['nama_lain']}*")
            if "penyebab" in konten:
                st.markdown(f"**Penyebab:** {konten['penyebab']}")

        st.divider()

        if "gejala_umum" in konten:
            st.subheader("Gejala Umum")
            st.markdown(konten["gejala_umum"])
        
        if "penanganan_dan_pengobatan" in konten:
            st.subheader("Penanganan dan Pengobatan")
            st.markdown(konten["penanganan_dan_pengobatan"])

        if "pencegahan" in konten:
            st.subheader("Tips Pencegahan")
            st.markdown(konten["pencegahan"])

# ======================
# ----- HALAMAN RIWAYAT -----
# ======================
elif page == "📝 Riwayat":
    st.title("📝 Riwayat Deteksi")
    st.markdown("Berikut adalah riwayat gambar yang pernah Anda deteksi.")

    try:
        files = sorted(
            [f for f in os.listdir(HISTORY_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))],
            reverse=True
        )
    except FileNotFoundError:
        files = []

    if not files:
        st.info("Belum ada riwayat deteksi.")
    else:
        JUMLAH_KOLOM = 4
        cols = st.columns(JUMLAH_KOLOM)

        for i, file_name in enumerate(files):
            with cols[i % JUMLAH_KOLOM]:
                try:
                    parts = file_name.split('_')
                    timestamp_str = f"{parts[0]}_{parts[1]}"
                    label_part = "_".join(parts[2:])
                    label = os.path.splitext(label_part)[0]
                    
                    dt_object = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    formatted_time = dt_object.strftime("%d %b %Y, %H:%M")
                
                except (ValueError, IndexError):
                    label = os.path.splitext(file_name)[0]
                    formatted_time = "Waktu tidak diketahui"

                st.markdown(f'<div class="card">', unsafe_allow_html=True)
                
                image_path = os.path.join(HISTORY_DIR, file_name)
                st.image(image_path, use_container_width=True)
                
                st.markdown(f"**Hasil:** `{label}`")
                st.caption(f"Waktu: {formatted_time}")
                
                if st.button("Hapus", key=file_name):
                    os.remove(image_path)
                    st.rerun() 

                st.markdown(f'</div>', unsafe_allow_html=True)

# ======================
# ----- HALAMAN TENTANG -----
# ======================
elif page == "ℹ️ Tentang":
    st.title("ℹ️ Tentang Aplikasi")
    st.markdown("""
        Aplikasi **IkanCheck** ini dikembangkan untuk membantu deteksi dini penyakit pada ikan air tawar menggunakan teknologi kecerdasan buatan.
        
        - **Model**: *Convolutional Neural Network* (CNN) dengan arsitektur Xception.
        - **Dataset**: Dilatih menggunakan dataset *'Fresh Water Fish Disease'* dari Kaggle.
        - **Framework**: Dibuat dengan menggunakan Streamlit.
        
        Semoga aplikasi ini dapat bermanfaat!
    """)