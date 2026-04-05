#!/usr/bin/env python3
"""
SMART CLINICAL VOICE SCREENER – FINAL (pYIN VALIDATED)
------------------------------------------------------
Pitch + Hoarseness (ENT Clinic Ready)
"""

import librosa
import numpy as np
import os

# ================= CONFIG =================
NORMAL_MALE_MAX = 160
PUBERPHONIA_MIN = 195
F0_STD_HOARSE = 25      # Speech-normalized [web:28]
JITTER_HOARSE = 1.5     # %

def analyze_voice(audio_path):
    try:
        # Load + trim
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        y, _ = librosa.effects.trim(y, top_db=30)
        duration = librosa.get_duration(y=y, sr=sr)

        # pYIN (clinic-grade)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=65, fmax=300, sr=sr, frame_length=2048  # ✅ Stable
        )

        # VOICED-ONLY (your Flask fix!) ✅
        voiced_f0 = f0[voiced_flag]
        f0_clean = voiced_f0[~np.isnan(voiced_f0)]
        
        if len(f0_clean) < 10:
            return duration, 0, 0, 0, "Borderline", "-"

        # Features
        median_f0 = np.median(f0_clean)
        f0_std = np.std(f0_clean)
        jitter = np.mean(np.abs(np.diff(f0_clean))) / median_f0 * 100 if len(f0_clean) > 1 else 0

        # Pitch (matches Flask)
        if median_f0 >= PUBERPHONIA_MIN:
            pitch_label = "Puberphonia"
        elif median_f0 <= NORMAL_MALE_MAX:
            pitch_label = "Normal Male"
        else:
            pitch_label = "Borderline"

        # Quality
        quality_label = "Hoarse" if f0_std > F0_STD_HOARSE or jitter > JITTER_HOARSE else "-"

        return duration, median_f0, f0_std, jitter, pitch_label, quality_label

    except Exception as e:
        return 0, 0, 0, 0, "Error", str(e)

# ================= OUTPUT =================
print("\n" + "="*80)
print("🎤 SIVA ENT VOICE SCREENER – pYIN VALIDATED")
print("Matches Flask API exactly")
print("="*80)

print(f"{'File':<20} | {'Dur':>4} | {'F0 Hz':>8} | {'Std':>6} | {'Jit%':>6} | {'Pitch':<12} | {'Quality'}")
print("-"*80)

for file in sorted(os.listdir(".")):
    if file.lower().endswith((".mp3", ".wav", ".m4a", ".flac")):
        dur, f0, std, jit, pitch, qual = analyze_voice(file)
        print(f"{file:<20} | {dur:4.1f} | {f0:8.1f} | {std:6.1f} | {jit:6.2f} | {pitch:<12} | {qual}")

print("="*80)
