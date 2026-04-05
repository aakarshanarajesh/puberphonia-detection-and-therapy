# 🎙️ Puberphonia Voice Analyzer & Therapy Assistant

## 🔍 Overview

Puberphonia is a voice disorder where individuals retain a high-pitched voice even after puberty. This project presents a **clinical voice screening system** that analyzes speech recordings to detect puberphonia and assess voice quality.

The system uses advanced pitch tracking (pYIN) and acoustic feature extraction to provide **automated screening and therapy guidance support**.

---

## 🎯 Problem Statement

Diagnosis of puberphonia typically requires clinical expertise and repeated evaluation. There is a need for an **accessible, automated tool** that can assist ENT specialists and patients in early detection and monitoring.

---

## 💡 Solution

This project implements a **Flask-based voice analysis API** that:

* Accepts audio input
* Extracts pitch and voice features
* Classifies voice condition
* Provides confidence-based results
* Stores clinical data for tracking

---

## 🚀 Features

* 🎤 Audio upload and processing
* 📊 Fundamental frequency (F0) detection using pYIN
* 📉 Pitch variability (standard deviation)
* 🔁 Jitter calculation for voice stability
* 🧠 Classification:

  * Normal Male
  * Borderline
  * Puberphonia
* 🎯 Confidence-based validation (eliminates false detection)
* 🏥 Voice quality detection (Clear / Hoarse)
* 💾 Automatic result storage (CSV + Excel)
* 🌐 REST API for frontend integration

---

## 🛠️ Tech Stack

* Python
* Flask (Backend API)
* Librosa (Audio Processing)
* NumPy, Pandas
* FFmpeg (Audio conversion)

---

## ⚙️ How It Works

1. User uploads voice recording
2. Audio is converted to WAV format
3. pYIN algorithm extracts pitch (F0)
4. Voiced frames are filtered using confidence
5. Features calculated:

   * Median F0
   * Standard deviation
   * Jitter (%)
6. Classification is performed based on thresholds
7. Results are returned and stored

---

## ▶️ How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python voice_screener_api.py
```

Then open:

```
http://localhost:5000
```

---

## 📊 Output

The system provides:

* Median Pitch (Hz)
* Pitch Variability
* Jitter (%)
* Voice Classification
* Confidence Level

Results are saved in:

* `voice_results.csv`
* `voice_results.xlsx`

---

## 🧠 Key Innovation

* ✅ Confidence-based pitch validation (avoids incorrect 90Hz fallback)
* ✅ Clinical threshold-based classification
* ✅ Real-time API-based voice screening
* ✅ Automatic data logging for medical tracking

---

## 📌 Applications

* ENT clinics
* Speech therapy support
* Voice disorder screening
* Remote health monitoring

---

## 🔮 Future Improvements

* Deep learning-based classification
* Mobile app integration
* Real-time feedback system
* Patient dashboard for tracking progress

---

## 👨‍💻 Author

**Aakarshana R**
CSE + Medical Engineering Student
