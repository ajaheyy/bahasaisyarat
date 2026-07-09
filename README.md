# SIBI Sign Language Detection System 🤟

Aplikasi web deteksi Bahasa Isyarat SIBI (A-Z) berbasis Flask dan YOLOv8 yang telah dioptimasi untuk berjalan super ringan di cloud platform (seperti Railway/Render free tier).

---

## 🚀 Mengapa Migrasi ke Format ONNX?

Sebelumnya, aplikasi ini menggunakan format model bawaan PyTorch (`.pt`). Saat dideploy ke Railway, ini memicu kendala besar:
1. **OOM (Out Of Memory) Crash:** Runtime PyTorch + Ultralytics memakan RAM lebih dari **1.2 GB** saat meload model. Di Railway Free/Hobby Tier (limit RAM 512MB), aplikasi akan langsung crash.
2. **Ukuran Build Raksasa:** Menginstal library `torch` dan `torchvision` membutuhkan download package sebesar **2 GB+**, membuat proses build sangat lambat dan rawan timeout.

**Solusi:**
Kita mengekspor model `my_model.pt` ke format **ONNX** (`my_model.onnx`) dan membacanya menggunakan modul **OpenCV DNN** (`cv2.dnn`).
* **Penggunaan RAM:** Turun drastis dari **1.2 GB** menjadi hanya **~70 MB**.
* **Ukuran Build:** Turun dari **3 GB+** menjadi hanya **~150 MB** (tidak butuh PyTorch lagi!).
* **Kecepatan Build:** Selesai dalam waktu kurang dari 1 menit di Railway.

---

## 🛠️ Optimasi yang Telah Diterapkan

Aplikasi ini telah melalui proses optimasi performa tinggi untuk kenyamanan pengguna:

1. **Buttery Smooth 60 FPS Camera (Render Lokal):**
   - Gambar dari kamera tidak lagi dikirim bolak-balik dari server untuk ditampilkan (yang menyebabkan lag parah).
   - Video webcam di-render secara lokal menggunakan loop `requestAnimationFrame` pada HTML5 Canvas langsung di browser di kecepatan **60 FPS** mulus tanpa lag.
2. **Transfer Data Super Ringan (JSON Coordinate Only):**
   - Server tidak mengirim ulang file gambar base64 hasil deteksi yang berukuran puluhan KB.
   - Server hanya mengirimkan koordinat pelacakan `[left, top, width, height]` dan label prediksi teks yang berukuran **< 100 bytes**, menghemat bandwidth internet secara ekstrem.
3. **Kompresi Resolusi Deteksi:**
   - Frame tangkapan kamera dikompresi ke resolusi **320x240** dengan kualitas JPEG **0.5** saat dikirim ke server. Ini meminimalkan ukuran payload upload menjadi hanya **~8 KB per request**.
4. **Mirror Kamera & Deteksi Sinkron:**
   - Kamera di-mirror secara horizontal agar terasa natural seperti cermin biasa.
   - Posisi bounding box secara otomatis di-mirror di sisi client agar pas dengan gerakan tangan, sedangkan label teks deteksi (A-Z) tetap digambar normal (tidak terbalik) agar mudah dibaca.
5. **Thread Safety Server:**
   - Proses inferensi model OpenCV DNN di server diamankan menggunakan `threading.Lock()` guna mencegah tabrakan data memori saat diakses oleh banyak pengguna secara bersamaan.
6. **Fix Jump Scroll:**
   - Memperbaiki tombol "Generate" pada bagian *Text to Gesture* agar tidak men-scroll halaman kembali ke paling atas saat diklik.

---

## 📋 Langkah Deploy ke Railway

### Prasyarat
- Akun GitHub yang terhubung dengan repositori ini.
- Akun Railway (yang sudah terverifikasi).

### Langkah-langkah:
1. Masuk ke **[Railway Dashboard](https://railway.app)**.
2. Klik **New Project** → **Deploy from GitHub Repo**.
3. Pilih repositori kamu (`bahasaisyarat`).
4. Railway akan otomatis mendeteksi konfigurasi `railway.toml` dan `Procfile` kita (menggunakan builder **Railpack**).
5. Proses instalasi dependensi (Python 3, OpenCV headless, Flask, Gunicorn) akan berjalan otomatis dan selesai dalam waktu **~1 menit**.
6. Setelah deployment bertuliskan **Active**, masuk ke tab **Settings** pada dashboard Railway layanan kamu.
7. Di bagian **Environment**, temukan sub-section **Networking**, lalu klik **Generate Domain** untuk mendapatkan URL publik aplikasi kamu.
8. Buka URL tersebut, dan aplikasi SIBI siap digunakan!

---

## 📦 Struktur Dependensi (`requirements.txt`)
Karena menggunakan OpenCV DNN, kita hanya memerlukan library Python minimal berikut:
```text
flask
gunicorn
opencv-python-headless
numpy
```
*(Tidak membutuhkan torch, torchvision, atau ultralytics lagi di server!)*
