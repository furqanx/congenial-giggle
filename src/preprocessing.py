import os
import torch
import librosa
import numpy as np
import tensorflow as tf
import noisereduce as nr
from audiomentations import AddBackgroundNoise

class DynamicNoiseInjector:
    def __init__(self, noise_dir="data/noise", min_snr_db=5, max_snr_db=15, p=0.5):
        """
        Inisialisasi penyuntik noise.
        Dibuat sebagai Class agar folder noise hanya dipindai 1 kali di awal (sangat cepat).
        
        Args:
            noise_dir: Path ke folder kumpulan file noise (.flac/.wav)
            min_snr_db & max_snr_db: Rentang rasio suara-ke-noise. Semakin kecil, noise semakin keras.
            p: Probabilitas (0.5 = 50% data akan diinjeksi noise, 50% dibiarkan bersih).
        """
        print(f"[Preprocessing] Menyiapkan amunisi noise dari: {noise_dir} (Probabilitas: {p*100}%)")
        self.augmentor = AddBackgroundNoise(
            sounds_path=noise_dir,
            min_snr_db=min_snr_db,
            max_snr_db=max_snr_db,
            p=p
        )

    def __call__(self, waveform_np: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Menyuntikkan noise secara acak ke dalam array audio.
        
        Args:
            waveform_np: 1D Numpy Array dari audio anak-anak yang bersih (Hasil dari torchaudio)
            sample_rate: Frekuensi sampel (standar 16000)
        """
        # audiomentations akan otomatis memilih 1 file noise secara acak dari folder,
        # menyesuaikan panjangnya, dan mencampurkannya dengan waveform kita.
        mixed_audio = self.augmentor(samples=waveform_np, sample_rate=sample_rate)
        return mixed_audio