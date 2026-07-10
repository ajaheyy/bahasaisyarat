# SIBI Sign Language Detection System 🤟

Aplikasi web deteksi Bahasa Isyarat SIBI (A-Z) berbasis Flask dan YOLOv8 yang menggunakan PyTorch dan Ultralytics, dioptimalkan untuk performa deteksi real-time tinggi dan berjalan secara efisien di cloud platform (seperti Railway).

---

## ⚡ Arsitektur Deteksi Real-Time & Cepat

Untuk mencapai deteksi yang terasa instan dan tidak patah-patah, aplikasi ini menggunakan pembagian beban kerja (pipeline separation) antara client dan server:

1. **Buttery Smooth 60 FPS Camera (Render Lokal):**
   - Video dari kamera **tidak dikirim bolak-balik** untuk digambar di server.
   - Video webcam di-render secara lokal di browser menggunakan loop `requestAnimationFrame` pada HTML5 Canvas di kecepatan **60 FPS** mulus.
2. **Transfer Data Minimal (Coordinate-Only):**
   - Server tidak mengirim ulang file gambar base64 hasil deteksi yang berat (menghemat bandwidth hingga **99%**).
   - Server hanya mengirimkan koordinat pelacakan `[left, top, width, height]` dan label teks prediksi (payload **< 100 bytes**).
3. **Mekanisme Self-Chaining Loop (Anti-Queue):**
   - Browser mengirim frame baru *hanya setelah* respon frame sebelumnya selesai diproses. Hal ini menghindari penumpukan antrean request jika koneksi internet pengguna mengalami fluktuasi latency.
   - Diberikan jeda throttling **30ms** untuk menjaga kestabilan load browser dan server.

---

## 🛠️ Optimasi Performa PyTorch di Server (CPU)

Menjalankan model PyTorch (.pt) di CPU server hosting gratisan seringkali berat dan menyebabkan lag. Kami menerapkan teknik optimasi berikut untuk mempercepat pemrosesan server:

1. **CPU Threading Control (`torch.set_num_threads(1)`):**
   - Mencegah *CPU contention/thread thrashing* dengan membatasi PyTorch hanya menggunakan 1 thread per proses eksekusi Flask.
2. **Image Size Reduction (`imgsz=224`):**
   - Frame webcam diperkecil menjadi **224x224** px sebelum dikirim dan diproses oleh YOLOv8. Ukuran data berkurang **~50%**, meningkatkan kecepatan inference CPU hingga **2x - 3x lipat**.
3. **Native Confidence Threshold:**
   - Menyaring deteksi langsung pada level C++ saat pemanggilan `model.predict(..., conf=0.4)` untuk menyingkirkan box tidak penting sebelum NMS (Non-Maximum Suppression) berjalan.
4. **Model Warmup pada Startup:**
   - Server melakukan deteksi dummy 1 frame saat aplikasi booting awal. Hal ini menghindari *cold start latency spike* (deteksi pertama yang lambat saat webcam pertama dinyalakan).
5. **Temporal Prediction Smoothing (Antiflicker):**
   - Browser menyimpan riwayat 3 prediksi terakhir dan menampilkan konsensus mayoritas untuk meminimalkan flicker/kedipan teks deteksi akibat noise frame sesaat.

---

## 📦 Dependensi Ringan CPU-Only (`requirements.txt`)

Kami mengonfigurasi instalasi PyTorch untuk menggunakan varian **CPU-only** (tanpa CUDA / GPU driver yang memakan memori bergiga-giga):
```text
flask
gunicorn
opencv-python-headless
numpy
ultralytics
--extra-index-url https://download.pytorch.org/whl/cpu
torch
torchvision
```
Ini menjaga ukuran build tetap ringkas (~150MB - 200MB) dan ramah terhadap limit memori container.

---

## 🐳 Konfigurasi Deploy di Railway

Karena library OpenCV membutuhkan beberapa dependencies sistem pada sistem operasi Linux, deployment menggunakan **Dockerfile** khusus:

1. **System Dependencies:**
   Dockerfile memasang package sistem yang dibutuhkan OpenCV seperti `libgl1`, `libglib2.0-0`, dan `libxcb1` untuk menghindari error `libxcb.so.1: cannot open shared object file`.
2. **Docker Builder:**
   Layanan dideploy menggunakan Railway Docker builder dengan konfigurasi di `railway.toml`:
   ```toml
   [build]
   builder = "dockerfile"
   dockerfilePath = "Dockerfile"
   ```
