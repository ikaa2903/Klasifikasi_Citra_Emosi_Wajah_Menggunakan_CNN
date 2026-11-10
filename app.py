import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

OUT_DIR = "output" if os.path.isdir("output") else "."
IMG_SIZE = (48, 48)

st.set_page_config(page_title="Klasifikasi Emosi Wajah CNN", layout="wide")

EMOJI = {"angry": "😠", "disgust": "🤢", "fear": "😨", "happy": "😄",
         "neutral": "😐", "sad": "😢", "surprise": "😲"}


# ------------------------------------------------------------- utilitas
def idn(x, d=2):
    """Format angka gaya Indonesia (koma desimal)."""
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def tampil(fig):
    st.pyplot(fig)
    plt.close(fig)


def _path(name):
    return os.path.join(OUT_DIR, name)


def show_image(fname):
    """Tampilkan gambar hasil notebook (folder output/). Tidak ada gambar yang dibuat ulang."""
    if os.path.exists(_path(fname)):
        st.image(_path(fname))
    else:
        st.info(f"{fname} belum ada di folder {OUT_DIR}/ (jalankan notebook sampai selesai).")


# ------------------------------------------------- ruang analisis manual
# Teks di bawah sudah diisi sesuai hasil notebook Anda (epoch terbaik 34, akurasi uji 71,65%, dst).
# Silakan edit langsung. Markdown boleh, multi-baris pakai """ ... """.
# Jika sebuah isian dikosongkan (""), dashboard memakai ANALISIS OTOMATIS dari data.
ANALISIS = {
    "ringkasan": """
- Dataset terdiri dari **7 kelas emosi** dengan total **49.779 gambar**.
- Seluruh gambar digabung lalu dibagi ulang secara *stratified* menjadi **39.823 gambar latih (80%)** dan **9.956 gambar uji (20%)**,
  sehingga proporsi tiap kelas di data latih dan data uji sama.
- Pada EDA, gambar asli berukuran 96x96 piksel, berwarna (RGB), format JPEG. Pada pra-pemrosesan gambar diubah menjadi
  **48x48 piksel grayscale (1 channel)** dan dinormalisasi ke rentang 0-1.
""",
    "distribusi": """
Berdasarkan grafik distribusi jumlah gambar per kelas emosi, terlihat bahwa kelas happy memiliki jumlah gambar terbanyak, yaitu 11.398 gambar, sedangkan kelas neutral sebanyak 8.166 gambar dan sad sebanyak 6.535 gambar. Sementara itu, kelas angry, disgust, fear, dan surprise masing-masing memiliki 5.920 gambar. Hal ini menunjukkan bahwa jumlah data pada setiap kelas emosi masih belum seimbang karena kelas happy memiliki jumlah gambar yang lebih banyak dibandingkan kelas lainnya. Kondisi ini perlu diperhatikan saat proses pelatihan model agar model tidak terlalu cenderung mengenali kelas tertentu saja.
""",
    "contoh_gambar": """
Contoh acak (6 gambar per kelas) menunjukkan variasi data yang tinggi: usia berbeda (bayi sampai dewasa), kepala miring, kacamata, wajah tertutup tangan,
serta pencahayaan gelap atau terang. Beberapa gambar tampak miring dengan sudut hitam di tepinya (terlihat pada beberapa contoh disgust dan fear),
dan ada gambar berkontras tinggi mirip sketsa pada kelas surprise. Ekspresi **sad, neutral,** dan **fear** pada beberapa contoh terlihat mirip,
sehingga sulit dibedakan. Variasi ini membuat klasifikasi menjadi menantang.
""",
    "kualitas_citra": """
Berdasarkan kedua histogram tersebut, distribusi kecerahan gambar paling banyak berada pada rentang 100–170, dengan puncak sekitar 130–140. Hal ini menunjukkan bahwa sebagian besar gambar memiliki tingkat kecerahan sedang hingga cukup terang. Sementara itu, distribusi kontras didominasi nilai 30–65, dengan puncak sekitar 40–50, yang menunjukkan bahwa sebagian besar gambar memiliki tingkat kontras sedang.
""",
    "contoh_latih": """
Contoh batch data latih setelah pra-pemrosesan: semua gambar sudah **48x48 piksel, grayscale**, dengan label emosi masing-masing
(diubah menjadi one-hot encoding saat pelatihan). Normalisasi 0-1 dan augmentasi (flip horizontal, rotasi, zoom, shift) dilakukan sebagai layer
di dalam model sehingga augmentasi hanya aktif saat pelatihan.
""",
    "akurasi": """
Berdasarkan tabel hasil pelatihan, akurasi training meningkat dari 0,3724 pada epoch pertama menjadi 0,7984 pada epoch ke-42. Sementara itu, akurasi validasi meningkat hingga mencapai nilai tertinggi sebesar 0,7165 atau 71,65% pada epoch ke-34, ke-38, dan ke-44. Hal ini menunjukkan bahwa model mampu mempelajari pola data dengan baik, tetapi peningkatan akurasi training tidak diikuti peningkatan akurasi validasi yang signifikan pada epoch akhir.
""",
    "loss": """
Berdasarkan nilai loss, training loss mengalami penurunan dari 1,7551 pada epoch pertama menjadi 0,5557 pada epoch ke-36. Sementara itu, validation loss mengalami fluktuasi pada awal pelatihan, kemudian menurun hingga mencapai nilai minimum sebesar 0,8096 pada epoch ke-30. Setelah itu, nilai validation loss cenderung meningkat dan stabil di kisaran 0,81. Kondisi ini menunjukkan adanya indikasi overfitting, karena model semakin baik mempelajari data training, tetapi performanya pada data validasi tidak mengalami peningkatan yang berarti.
""",
    "metrik": """
- **Accuracy 71,65%**: sekitar 72 dari 100 gambar uji diprediksi benar.
- **Macro average**: Precision 0,698, Recall 0,703, F1-score 0,700.
- **Weighted average**: Precision 0,717, Recall 0,716, F1-score 0,716. Nilai weighted sedikit lebih tinggi karena kelas happy
  (data uji terbanyak, 2.280) memiliki performa terbaik.
""",
    "metrik_kelas": """
- **Kelas terbaik: happy** (F1 0,879), lalu disgust (0,835) dan surprise (0,776). **Kelas terlemah: fear** (F1 0,552), lalu sad (0,583) dan angry (0,619).
- Precision happy (0,897) lebih tinggi daripada recall-nya (0,862): jika model menyebut happy biasanya benar, tetapi sebagian gambar happy terlewat.
  Pada disgust sebaliknya, recall (0,865) lebih tinggi dari precision (0,808).
- Jumlah data bukan satu-satunya penentu: disgust dan angry sama-sama 5.920 gambar, tetapi F1 disgust (0,835) jauh di atas angry (0,619),
  sehingga kemiripan ekspresi antar kelas ikut berperan.
""",
    "confusion_matrix": """
Berdasarkan grafik Confusion Matrix, model mampu mengklasifikasikan beberapa kelas emosi dengan cukup baik, terutama kelas happy dengan 1.966 prediksi benar, disgust sebanyak 1.024, dan surprise sebanyak 947. Sementara itu, kelas angry, fear, neutral, dan sad masih mengalami kesalahan prediksi, terutama kelas neutral yang sering diprediksi sebagai sad sebanyak 198 gambar. Selain itu, kelas fear juga cukup sering tertukar dengan sad sebanyak 123 gambar dan surprise sebanyak 134 gambar. Hal ini menunjukkan bahwa model masih kesulitan membedakan beberapa emosi yang memiliki karakteristik ekspresi wajah yang mirip. Secara keseluruhan, model sudah mampu mengenali sebagian besar kelas emosi dengan baik, tetapi masih perlu ditingkatkan agar kesalahan klasifikasi antar kelas dapat dikurangi.
""",
    "contoh_prediksi": """
Dari 12 contoh acak data uji, **8 diprediksi benar (hijau)** dan **4 salah (merah)**: sad diprediksi angry (90,8%), sad diprediksi angry (94,8%),
neutral diprediksi surprise (72,4%), dan sad diprediksi fear (70,5%). Ini sejalan dengan confusion matrix (sad dan neutral sering salah).
Beberapa prediksi salah memiliki confidence tinggi (di atas 90%), sedangkan beberapa prediksi benar confidence-nya rendah
(sad 41,8%, neutral 54,3%), jadi confidence tidak selalu menjamin prediksi benar.
""",
    "gradcam": """
Grad-CAM menampilkan area yang paling memengaruhi keputusan model (merah/kuning = pengaruh tinggi). Dari 8 contoh, **5 diprediksi benar** dan **3 salah**
(sad → angry dua kali, neutral → surprise). Pada beberapa prediksi benar (mis. disgust dan sad), area panas berada di sekitar **mulut, alis, dan mata**.
Pada contoh yang salah, area panas cenderung tersebar atau jatuh pada bagian kurang informatif (mis. dahi/tepi wajah pada gambar gelap).
Ini interpretasi dari sampel kecil, bukan bukti pasti untuk seluruh data.
""",
    "prediksi": "",
    "gradcam_upload": "",
}


def analisis(kunci, auto=None):
    """Tampilkan ANALISIS[kunci]; jika kosong pakai analisis otomatis (daftar kalimat)."""
    teks = ANALISIS.get(kunci, "").strip()
    with st.container(border=True):
        if teks:
            st.markdown("**Analisis:**")
            st.markdown(teks)
        elif auto:
            st.markdown("**Analisis:**")
            st.markdown("\n".join(f"- {l}" for l in auto if l))
            st.caption(f"(analisis otomatis — tulis di `ANALISIS[\"{kunci}\"]` pada app.py untuk menggantinya)")
        else:
            st.markdown("**Analisis:**")
            st.caption(f"(belum diisi — tulis di `ANALISIS[\"{kunci}\"]` pada app.py)")
            st.write("")
            st.write("")


# ------------------------------------------------------- data & model
@st.cache_data
def load_json(name):
    p = _path(name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_model(name):
    import tensorflow as tf

    return tf.keras.models.load_model(_path(name), compile=False)


summary = load_json("dataset_summary.json")
history = load_json("history.json")
config = load_json("training_config.json")
metrics = load_json("eval_metrics.json")
cm_data = load_json("confusion_matrix.json")
class_names = load_json("class_names.json") or (summary or {}).get("class_names")

# ---------------------------------------------------- judul & deskripsi
st.title("Klasifikasi Citra Emosi Wajah Menggunakan CNN")
st.markdown(
    "Dashboard ini menampilkan hasil eksperimen **Convolutional Neural Network (CNN)** untuk "
    "mengenali emosi dari citra wajah. Data berasal dari "
    "[Facial Emotion Recognition Dataset (Kaggle)]"
    "(https://www.kaggle.com/datasets/fahadullaha/facial-emotion-recognition-dataset/data). "
    "Alur: ringkasan dataset → distribusi data → pra-pemrosesan (resize 48×48 grayscale, "
    "normalisasi 0–1, split 80:20, one-hot) → pelatihan CNN → evaluasi (Accuracy, Precision, "
    "Recall, F1-score, confusion matrix) → uji prediksi gambar baru."
)

if not os.path.isdir(OUT_DIR) or not class_names:
    st.error(f"Folder `{OUT_DIR}/` atau hasil notebook tidak ditemukan. Jalankan notebook sampai "
             "selesai, lalu jalankan `streamlit run app.py` dari folder yang sama dengan `output/`.")
    st.stop()

# ---------------------------------------------------------- sidebar
st.sidebar.header("Pengaturan")
model_files = [f for f in ("emotion_cnn_best.keras", "emotion_cnn_model.keras") if os.path.exists(_path(f))]
model_file = st.sidebar.selectbox("Model untuk prediksi", model_files) if model_files else None
show_cam = st.sidebar.checkbox("Tampilkan Grad-CAM pada prediksi", value=True)
st.sidebar.markdown("---")
st.sidebar.caption(f"{len(class_names)} kelas: " + ", ".join(class_names))

tab_ringkas, tab_dist, tab_train, tab_eval, tab_pred = st.tabs([
    "1. Ringkasan Dataset", "2. Distribusi Data", "3. Hasil Pelatihan",
    "4. Evaluasi & Confusion Matrix", "5. Uji Prediksi",
])


# ------------------------------------------------ analisis otomatis
# (dipakai hanya jika isian di ANALISIS dikosongkan)
def auto_split():
    s = summary
    return [
        f"Dataset memuat **{idn(s['total_images'], 0)} gambar** dalam **{s['num_classes']} kelas** emosi "
        f"({', '.join(s['class_names'])}); ukuran input model {s['image_size'][0]}×{s['image_size'][1]} piksel, {s['color_mode']}.",
        f"Data dibagi **{s['train_ratio']:.0%} latih ({idn(s['train_images'], 0)})** dan "
        f"**{s['val_ratio']:.0%} uji ({idn(s['val_images'], 0)})** secara stratified, sehingga proporsi "
        "tiap kelas sama di kedua set.",
    ]


def auto_dist():
    c = summary["class_counts"]
    mx, mn = max(c, key=c.get), min(c, key=c.get)
    r = summary["imbalance_ratio"]
    out = [f"Kelas terbanyak **{mx}** ({idn(c[mx], 0)} gambar, {c[mx] / sum(c.values()) * 100:.1f}%), "
           f"tersedikit **{mn}** ({idn(c[mn], 0)}). Rasio ketidakseimbangan {idn(r)}×."]
    if r > 1.5:
        out.append(f"Data **tidak seimbang**; model berpotensi lebih condong ke kelas mayoritas (**{mx}**), "
                   "sehingga F1/recall per kelas perlu dilihat selain akurasi total.")
    else:
        out.append("Distribusi relatif seimbang.")
    return out


def auto_acc():
    h = history
    b = int(np.argmax(h["val_accuracy"]))
    ta, va = h["accuracy"][b], h["val_accuracy"][b]
    gap = ta - va
    out = [f"Training berjalan **{len(h['loss'])} epoch**; val_accuracy tertinggi **{va * 100:.2f}%** pada epoch "
           f"**{b + 1}** (train accuracy {ta * 100:.2f}%)."]
    if gap > 0.10:
        out.append(f"Selisih train–validasi {gap * 100:.1f} poin: ada **indikasi overfitting**.")
    elif gap < -0.02:
        out.append(f"Akurasi validasi {abs(gap) * 100:.1f} poin lebih tinggi dari train.")
    else:
        out.append(f"Selisih train–validasi {gap * 100:.1f} poin.")
    if len(h["loss"]) > b + 1:
        out.append(f"Setelah epoch {b + 1} val_accuracy tidak membaik; EarlyStopping menghentikan training "
                   "dan mengembalikan bobot terbaik.")
    return out


def auto_loss():
    h = history
    vl = h["val_loss"]
    b = int(np.argmin(vl))
    out = [f"Loss latih turun dari {idn(h['loss'][0], 3)} menjadi {idn(h['loss'][-1], 3)}; val_loss minimum "
           f"**{idn(vl[b], 3)}** pada epoch {b + 1}."]
    if vl[-1] > min(vl) * 1.05:
        out.append("Val_loss naik di akhir training, tanda model mulai overfit pada data latih.")
    else:
        out.append("Val_loss stabil/menurun sampai akhir, tidak tampak overfitting berarti.")
    return out


def auto_metrik():
    rep, acc = metrics["report"], metrics["accuracy"]
    m, wt = rep["macro avg"], rep["weighted avg"]
    out = [f"**Accuracy {acc * 100:.2f}%**: sekitar {acc * 100:.0f} dari 100 gambar uji ditebak benar.",
           f"**Precision macro {idn(m['precision'], 3)}**, **Recall macro {idn(m['recall'], 3)}**, "
           f"**F1 macro {idn(m['f1-score'], 3)}**."]
    if abs(wt["f1-score"] - m["f1-score"]) > 0.01:
        lbl = "lebih baik" if wt["f1-score"] > m["f1-score"] else "lebih buruk"
        out.append(f"F1 weighted ({idn(wt['f1-score'], 3)}) berbeda dari F1 macro ({idn(m['f1-score'], 3)}): "
                   f"performa pada kelas besar {lbl} dibanding kelas kecil.")
    return out


def auto_metrik_kelas():
    rep = metrics["report"]
    f1 = {c: rep[c]["f1-score"] for c in class_names}
    b, w = max(f1, key=f1.get), min(f1, key=f1.get)
    out = [f"Kelas terbaik **{b}** (F1 {idn(f1[b], 3)}); terlemah **{w}** (F1 {idn(f1[w], 3)}).",
           "Urutan F1: " + " > ".join(f"{c} ({idn(f1[c], 3)})" for c in sorted(f1, key=f1.get, reverse=True)) + "."]
    return out


def auto_cm(cm):
    rec = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    bi, wi = int(rec.argmax()), int(rec.argmin())
    pairs = sorted([(int(cm[i, j]), i, j) for i in range(len(cm)) for j in range(len(cm)) if i != j],
                   reverse=True)[:3]
    return [
        f"{idn(cm.diagonal().sum(), 0)} dari {idn(cm.sum(), 0)} gambar uji diprediksi benar "
        f"({cm.diagonal().sum() / cm.sum() * 100:.2f}%).",
        f"Paling akurat **{class_names[bi]}** ({rec[bi] * 100:.1f}% benar); paling sering salah "
        f"**{class_names[wi]}** (hanya {rec[wi] * 100:.1f}% benar).",
        "Kesalahan terbesar: " + "; ".join(
            f"**{class_names[i]} → {class_names[j]}** ({n} gambar)" for n, i, j in pairs) + ".",
    ]


def auto_pred(probs, i1, i2):
    top, sec = class_names[i1], class_names[i2]
    p1, p2 = probs[i1], probs[i2]
    out = [f"Model memprediksi **{top}** {EMOJI.get(top, '')} dengan keyakinan **{p1 * 100:.1f}%**; "
           f"kandidat kedua **{sec}** ({p2 * 100:.1f}%)."]
    if p1 >= 0.8:
        out.append("Keyakinan **tinggi**.")
    elif p1 >= 0.5:
        out.append(f"Keyakinan **sedang**: selisih dengan {sec} hanya {(p1 - p2) * 100:.1f} poin, "
                   "ekspresi mungkin ambigu.")
    else:
        out.append("Keyakinan **rendah**: probabilitas tersebar di beberapa kelas, hasil jangan dipercaya penuh. "
                   "Coba foto frontal, jelas, dan pencahayaan merata.")
    if metrics:
        f1 = metrics["report"][top]["f1-score"]
        out.append(f"Pada data uji, F1 kelas **{top}** = {idn(f1)} "
                   + ("(kelas yang andal dikenali)." if f1 >= 0.75
                      else "(kelas sulit, hasil perlu hati-hati)." if f1 < 0.6 else "(performa menengah)."))
    out.append(f"Akurasi model pada data uji {idn(metrics['accuracy'] * 100)}%, jadi prediksi pada gambar baru tetap bisa salah."
               if metrics else "")
    return out


# ===================================================== 1. RINGKASAN
with tab_ringkas:
    if summary:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Jumlah kelas", summary["num_classes"])
        c2.metric("Jumlah gambar", idn(summary["total_images"], 0))
        c3.metric("Data latih", f"{idn(summary['train_images'], 0)} ({summary['train_ratio']:.0%})")
        c4.metric("Data uji", f"{idn(summary['val_images'], 0)} ({summary['val_ratio']:.0%})")

        st.markdown(f"- Ukuran citra input model: **{summary['image_size'][0]}×{summary['image_size'][1]}** piksel\n"
                    f"- Mode warna: **{summary['color_mode']}**\n"
                    f"- Kelas: {', '.join(class_names)}")
        analisis("ringkasan", auto_split())

    st.subheader("Contoh gambar per kelas")
    show_image("2_sample_per_class.png")
    analisis("contoh_gambar", [
        "Setiap baris adalah sampel satu kelas emosi.",
        "Terlihat variasi dalam satu kelas (pencahayaan, sudut, usia, aksesori) dan kemiripan antar kelas.",
    ])

    with st.expander("Kualitas citra & contoh data latih"):
        show_image("3_image_quality.png")
        analisis("kualitas_citra", [
            "Histogram kecerahan dan kontras menunjukkan sebaran kualitas gambar pada sampel acak.",
        ])
        show_image("4_sample_training.png")
        analisis("contoh_latih", [
            "Data latih setelah pra-pemrosesan: seragam 48×48, 1 channel grayscale, label one-hot.",
        ])

# ===================================================== 2. DISTRIBUSI
with tab_dist:
    if summary:
        st.subheader("Jumlah gambar per kelas")
        show_image("1_class_distribution.png")

        df_c = pd.DataFrame({"Latih": summary["class_counts_train"],
                             "Uji": summary["class_counts_val"]})
        df_c["Total"] = df_c["Latih"] + df_c["Uji"]
        df_c["Persen (%)"] = (df_c["Total"] / df_c["Total"].sum() * 100).round(2)
        st.dataframe(df_c)
        analisis("distribusi", auto_dist())
        if summary["imbalance_ratio"] > 1.5:
            st.warning(f"Terdeteksi class imbalance (rasio {idn(summary['imbalance_ratio'])}×).")

# ===================================================== 3. PELATIHAN
with tab_train:
    if history:
        df_h = pd.DataFrame(history)
        df_h.index = range(1, len(df_h) + 1)
        best = int(np.argmax(history["val_accuracy"])) + 1

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Epoch dijalankan", len(df_h))
        c2.metric("Epoch terbaik", best)
        c3.metric("Val accuracy terbaik", f"{idn(df_h['val_accuracy'].max() * 100)}%")
        c4.metric("Val loss minimum", idn(df_h["val_loss"].min(), 4))

        st.subheader("Grafik akurasi dan loss")
        show_image("5_training_curves.png")
        st.markdown("**Grafik akurasi (kiri)**")
        analisis("akurasi", auto_acc())
        st.markdown("**Grafik loss (kanan)**")
        analisis("loss", auto_loss())

        with st.expander("Riwayat per epoch"):
            st.dataframe(df_h.round(4))
    else:
        st.info("history.json belum ada. Jalankan Langkah 4 di notebook.")

    if config:
        st.subheader("Konfigurasi model")
        st.dataframe(pd.Series({k: str(v) for k, v in config.items()}, name="Nilai"))
       

# ===================================================== 4. EVALUASI
with tab_eval:
    if not metrics:
        st.info("eval_metrics.json belum ada. Jalankan Langkah 5 di notebook.")
    else:
        rep = metrics["report"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{idn(metrics['accuracy'] * 100)}%")
        c2.metric("Precision (macro)", idn(rep["macro avg"]["precision"], 4))
        c3.metric("Recall (macro)", idn(rep["macro avg"]["recall"], 4))
        c4.metric("F1-score (macro)", idn(rep["macro avg"]["f1-score"], 4))
        analisis("metrik", auto_metrik())

        st.subheader("Precision, Recall, F1-score per kelas")
        df_r = pd.DataFrame(rep).T.loc[class_names + ["macro avg", "weighted avg"]]
        df_r = df_r.rename(columns={"precision": "Precision", "recall": "Recall",
                                    "f1-score": "F1-Score", "support": "Jumlah data"})
        df_r["Jumlah data"] = df_r["Jumlah data"].astype(int)
        st.dataframe(df_r.round(3))
        analisis("metrik_kelas", auto_metrik_kelas())

    st.subheader("Confusion matrix")
    show_image("6_confusion_matrix.png")
    if cm_data:
        cm = np.array(cm_data["matrix"])
        analisis("confusion_matrix", auto_cm(cm))

    with st.expander("Contoh hasil klasifikasi & Grad-CAM"):
        show_image("7_sample_predictions.png")
        analisis("contoh_prediksi", [
            "Judul hijau = prediksi benar, merah = salah, disertai keyakinan model.",
        ])
        show_image("8_gradcam.png")
        analisis("gradcam", [
            "Grad-CAM menyorot area (merah/kuning) yang paling memengaruhi keputusan model.",
        ])


# ===================================================== 5. PREDIKSI
def _load_cascade():
    try:
        casc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        return None if casc.empty() else casc
    except Exception:  # noqa: BLE001
        return None


_face_cascade = _load_cascade()


def preprocess(img):
    """
    Pra-pemrosesan gambar upload agar mirip data latih:
    1. Koreksi orientasi foto (EXIF) dan ubah ke grayscale.
    2. Deteksi wajah (Haar cascade), crop wajah terbesar + margin kecil.
       Jika tidak ada wajah terdeteksi, seluruh gambar dipakai.
    3. Resize ke 48x48 dengan INTER_AREA (anti-aliasing untuk downscale).
    4. Pertahankan skala piksel 0-255 karena model sudah punya
       layer Rescaling(1./255) di dalam arsitekturnya.
    """
    img = ImageOps.exif_transpose(img).convert("L")
    gray = np.asarray(img, dtype=np.uint8)

    faces = ()
    if _face_cascade is not None:
        faces = _face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        m = int(0.05 * w)
        y0, y1 = max(y - m, 0), min(y + h + m, gray.shape[0])
        x0, x1 = max(x - m, 0), min(x + w + m, gray.shape[1])
        gray = gray[y0:y1, x0:x1]
    else:
        st.warning("Wajah tidak terdeteksi, memakai seluruh gambar (hasil kurang akurat).")

    gray = cv2.resize(gray, IMG_SIZE, interpolation=cv2.INTER_AREA)

    # Bentuk akhir: (1, 48, 48, 1), rentang 0-255.
    return gray.astype(np.float32)[np.newaxis, ..., np.newaxis]


def gradcam(model, batch):
    import tensorflow as tf

    gm = tf.keras.models.Model(model.inputs, [model.get_layer("last_conv").output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = gm(batch, training=False)
        score = preds[:, tf.argmax(preds[0])]
    grads = tape.gradient(score, conv_out)
    w = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv_out * w[:, None, None, :], axis=-1)[0])
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return tf.image.resize(cam[..., None], IMG_SIZE).numpy().squeeze()


with tab_pred:
    if model_file is None:
        st.error("File model (.keras) tidak ditemukan di folder output/.")
    else:
        model = load_model(model_file)
        st.caption("Gunakan foto wajah close-up (wajah memenuhi gambar) agar mirip data latih. "
                   "Prediksi muncul otomatis begitu gambar dipilih.")
        src = st.radio("Sumber gambar", ["Upload", "Kamera"], horizontal=True)
        file = (st.file_uploader("Unggah gambar wajah", type=["jpg", "jpeg", "png", "bmp"])
                if src == "Upload" else st.camera_input("Ambil foto"))

        if file is not None:
            img = Image.open(file)
            batch = preprocess(img)
            with st.spinner("Memprediksi..."):
                probs = model.predict(batch, verbose=0)[0]
            order = np.argsort(probs)[::-1]
            i1, i2 = int(order[0]), int(order[1])
            top = class_names[i1]

            c1, c2, c3 = st.columns([1, 1, 1.4])
            c1.image(img, caption="Gambar asli")
            c2.image(Image.fromarray(batch[0].squeeze().astype("uint8")).resize((240, 240), Image.NEAREST),
                     caption="Input model (48×48 grayscale)")
            with c3:
                st.metric("Hasil prediksi", f"{EMOJI.get(top, '')} {top.upper()}",
                          f"{idn(probs[i1] * 100, 1)}% yakin")

            st.subheader("Probabilitas tiap emosi")
            fig, ax = plt.subplots(figsize=(8, 3.5))
            colors = ["tab:red" if i == i1 else "tab:blue" for i in range(len(class_names))]
            ax.bar(class_names, probs * 100, color=colors)
            ax.set_ylabel("Probabilitas (%)")
            ax.set_ylim(0, 100)
            for i, v in enumerate(probs * 100):
                ax.text(i, v + 1, f"{v:.1f}", ha="center", fontsize=9)
            tampil(fig)
            analisis("prediksi", auto_pred(probs, i1, i2))

            if show_cam:
                import matplotlib

                try:
                    cam = gradcam(model, batch)
                    gray = batch[0].squeeze() / 255.0
                    heat = matplotlib.colormaps["jet"](cam)[..., :3]
                    mix = (0.5 * np.stack([gray] * 3, -1) + 0.5 * heat).clip(0, 1)
                    st.subheader("Grad-CAM")
                    st.image(Image.fromarray((mix * 255).astype("uint8")).resize((240, 240), Image.NEAREST),
                             caption="Merah/kuning = area paling berpengaruh")
                    analisis("gradcam_upload", [
                        "Area merah/kuning adalah bagian wajah yang paling menentukan prediksi.",
                    ])
                except Exception as e:  # noqa: BLE001
                    st.warning(f"Grad-CAM gagal dibuat: {e}")
