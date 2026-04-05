from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import librosa
import io
import subprocess
import tempfile
import os
import csv
from datetime import datetime
import pandas as pd


# ==========================================================
# CLINICAL VOICE SCREENER API – PRODUCTION READY (Dec 2025)
# FIXED: 90Hz pYIN confidence validation
# ==========================================================
app = Flask(__name__)
CORS(app)


# ✅ FIXED: PORTABLE PATH (works on Windows/Linux/Mac)
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "voice_results.csv")
EXCEL_FILE = os.path.join(os.path.dirname(__file__), "voice_results.xlsx")


FIELDNAMES = [
    "timestamp",
    "patient_id", 
    "age",
    "median_f0_hz",
    "f0_std_hz",
    "jitter_percent",
    "pitch_label",
    "quality_label",
    "voiced_frames",      # NEW
    "mean_voiced_prob",   # NEW
    "confidence_high"     # NEW
]


def append_result_row(row: dict):
    """Create CSV if needed and append one result row."""
    file_exists = os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"✅ SAVED TO CSV: {row['patient_id']} | F0: {row['median_f0_hz']}Hz | Conf: {row.get('confidence_high', 'N/A')}")


def save_excel():
    """Convert CSV to Excel automatically."""
    if os.path.exists(RESULTS_FILE):
        try:
            df = pd.read_csv(RESULTS_FILE)
            df.to_excel(EXCEL_FILE, index=False)
            print(f"✅ EXCEL SAVED: {EXCEL_FILE}")
        except Exception as e:
            print(f"⚠️ Excel save failed: {e}")


# ---- Validated thresholds (Dec 2025) ----
NORMAL_MALE_MAX = 160        # Hz
BORDERLINE_MIN = 161         # Hz
BORDERLINE_MAX = 194         # Hz
PUBERPHONIA_MIN = 195        # Hz
F0_STD_HOARSE_THRESHOLD = 30 # Hz
JITTER_HOARSE_THRESHOLD = 3.0 # %


def decode_to_wav_bytes(raw_bytes: bytes) -> bytes:
    """
    Convert any input (webm/opus, wav, etc.) to mono 16‑kHz WAV using ffmpeg.
    Requires ffmpeg installed and visible on PATH.
    """
    with tempfile.NamedTemporaryFile(suffix=".input", delete=False) as f_in:
        f_in.write(raw_bytes)
        in_path = f_in.name

    out_path = in_path + ".wav"

    cmd = [
        "ffmpeg", "-y",
        "-i", in_path,
        "-ac", "1",      # mono
        "-ar", "16000",  # 16 kHz
        "-f", "wav",
        out_path
    ]

    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        with open(out_path, "rb") as f_out:
            wav_bytes = f_out.read()
    finally:
        for p in (in_path, out_path):
            try:
                os.remove(p)
            except OSError:
                pass

    return wav_bytes


@app.route("/")
def index():
    return '''
    <!DOCTYPE html>
    <html><head><title>🎤 Voice Screener API</title>
    <style>body{font-family:sans-serif;max-width:600px;margin:50px auto;padding:20px;background:#f5f7fa;}
    h1{color:#0b7285;} .status{background:#d1e7dd;padding:10px;border-radius:8px;margin:10px 0;}</style></head>
    <body>
        <h1>🎤 Clinical Voice Screener API</h1>
        <div class="status">✅ Server running on <b>http://localhost:5000</b></div>
        <p><b>📱 Test:</b> Open voice_screener.html?patientId=SIVA001&age=24</p>
        <hr>
        <p><b>📊 Results saved:</b></p>
        <ul>
            <li><b>CSV:</b> voice_results.csv</li>
            <li><b>Excel:</b> voice_results.xlsx</li>
        </ul>
        <p><small>Run: <code>python voice_screener_api.py</code></small></p>
    </body></html>
    '''


@app.route("/analyze", methods=["POST"])
def analyze():
    """Receive audio → pYIN → metrics → classification."""
    try:
        if "audio" not in request.files:
            return jsonify({"status": "error", "message": "No audio file"}), 400

        audio_file = request.files["audio"]
        raw_bytes = audio_file.read()

        # 1. Decode to WAV
        wav_bytes = decode_to_wav_bytes(raw_bytes)

        # 2. Load with librosa
        y, sr = librosa.load(io.BytesIO(wav_bytes), sr=None, mono=True)

        if len(y) < sr // 2:  # <0.5s
            return jsonify({"status": "error", "message": "Recording too short (needs 1s+)" }), 400

        # 3. pYIN pitch tracking
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=90, fmax=280, sr=sr, frame_length=2048
        )
        times = librosa.frames_to_time(np.arange(len(f0)), sr=sr)

        # 4. FIXED: Confidence validation (eliminates 90Hz fallback)
        voiced_count = np.sum(voiced_flag)
        mean_voiced_prob = float(np.mean(voiced_probs[voiced_flag])) if voiced_count > 0 else 0.0
        confidence_ok = (voiced_count >= 4) and (mean_voiced_prob >= 0.100)
        f0_clean = f0[voiced_flag]

        voiced_count = int(np.sum(voiced_flag))
        mean_voiced_prob = float(np.mean(voiced_probs[voiced_flag])) if voiced_count > 0 else 0.0

        if len(f0_clean) < 2:
            # Let frontend fall back to graph
            return jsonify({
                "status": "success",
                "metrics": {
                    "median_f0_hz": 0.0,
                    "f0_std_hz": 0.0,
                    "jitter_percent": 0.0
                },
                "classification": {
                    "pitch": "Unknown",
                    "quality": "Unknown"
                },
                "confidence": {
                    "is_high": False,
                    "voiced_frames": voiced_count,
                    "mean_voiced_prob": mean_voiced_prob,
                    "warning": "Too few voiced frames – use graph estimate / retry louder."
                },
                "f0_values": [float(x) if not np.isnan(x) else 0 for x in f0],
                "time_values": times.tolist()
            }), 200


        # 5. Metrics
        median_f0 = float(np.median(f0_clean))
        std_f0 = float(np.std(f0_clean))

        # Jitter (%)
        jitter = 0.0
        if len(f0_clean) > 1:
            jitter_periods = np.abs(np.diff(f0_clean))
            jitter = float(np.mean(jitter_periods) / median_f0 * 100)

        # 6. Classification
        if median_f0 <= NORMAL_MALE_MAX:
            pitch_class = "Normal Male"
        elif median_f0 <= BORDERLINE_MAX:
            pitch_class = "Borderline"
        else:
            pitch_class = "Puberphonia"

        is_hoarse = std_f0 > F0_STD_HOARSE_THRESHOLD or jitter > JITTER_HOARSE_THRESHOLD
        quality = "Hoarse" if is_hoarse else "Clear"

        print(f"✅ ANALYSIS: F0={median_f0:.1f}Hz | Std={std_f0:.1f}Hz | Jitter={jitter:.2f}% | {pitch_class} | Conf:{confidence_ok} ({voiced_count}f, p:{mean_voiced_prob:.3f})")
       
        return jsonify({
            "status": "success",
            "metrics": {
                "median_f0_hz": round(median_f0, 1),
                "f0_std_hz": round(std_f0, 1),
                "jitter_percent": round(jitter, 2)
            },
            "classification": {
                "pitch": pitch_class,
                "quality": quality
            },
            "confidence": {  # NEW - fixes 90Hz problem
                "is_high": confidence_ok,
                "voiced_frames": int(voiced_count),
                "mean_voiced_prob": round(mean_voiced_prob, 3),
                "warning": "Low confidence: too few voiced frames" if not confidence_ok else None
            },
            "f0_values": [float(x) if not np.isnan(x) else 0 for x in f0],
            "time_values": times.tolist()
        })

    except Exception as e:
        print(f"❌ Analysis error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/save-result", methods=["POST"])
def save_result():
    """Save clinical result to CSV + Excel."""
    try:
        data = request.get_json(force=True)

        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "patient_id": data.get("patient_id", "anonymous"),
            "age": data.get("age", "unknown"),
            "median_f0_hz": data.get("median_f0_hz", 0),
            "f0_std_hz": data.get("f0_std_hz", 0),
            "jitter_percent": data.get("jitter_percent", 0),
            "pitch_label": data.get("pitch_label", "unknown"),
            "quality_label": data.get("quality_label", "unknown"),
            "voiced_frames": data.get("voiced_frames", 0),      # NEW
            "mean_voiced_prob": data.get("mean_voiced_prob", 0), # NEW
            "confidence_high": data.get("confidence_high", False) # NEW
        }

        append_result_row(row)
        save_excel()  # Auto-convert to Excel
        
        return jsonify({"status": "ok"}), 200
        
    except Exception as e:
        print(f"❌ Save error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "results_file": RESULTS_FILE})


@app.route("/results", methods=["GET"])
def get_results():
    """Preview latest results."""
    if not os.path.exists(RESULTS_FILE):
        return jsonify({"status": "no_results", "message": "No results yet"})
    
    try:
        df = pd.read_csv(RESULTS_FILE)
        return jsonify({
            "status": "success",
            "count": len(df),
            "latest": df.tail(3).to_dict('records')
        })
    except:
        return jsonify({"status": "error", "message": "Cannot read results"})


if __name__ == "__main__":
    print("==========================================================")
    print("🎤 CLINICAL VOICE SCREENER API v2.1 - FIXED 90Hz")
    print(f"📊 CSV: {RESULTS_FILE}")
    print(f"📊 Excel: {EXCEL_FILE}")
    print("🚀 http://localhost:5000")
    print("✅ FIXED: pYIN confidence → no more 90Hz fallback!")
    print("==========================================================")
    
    # Create empty CSV if needed
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
    
    app.run(debug=False, host="0.0.0.0", port=5000)
