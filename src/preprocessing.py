import os
import numpy as np
import noisereduce as nr
import librosa
import tensorflow as tf
import torch

# ==========================================
#         CORE FUNCTIONS (NumPy Based)
# ==========================================

def _load_audio(file_name, target_sr):
    """Core logic: Load audio ke Numpy Array"""
    if isinstance(file_name, bytes):
        file_name = file_name.decode('utf-8')
    elif isinstance(file_name, tf.Tensor) and file_name.dtype == tf.string:
        file_name = file_name.numpy().decode('utf-8')
    
    try:
        waveform_np, _ = librosa.load(file_name, sr=target_sr, mono=True)
        return waveform_np.astype(np.float32)
    except Exception as e:
        print(f"WARNING: Gagal memuat {file_name}. Error: {e}")
        return np.zeros(1, dtype=np.float32)

def _trim_audio(waveform_np, top_db=20):
    """Core logic: Trim silence"""
    trimmed, _ = librosa.effects.trim(waveform_np, top_db=top_db)
    return trimmed

def _reduce_noise(waveform_np, sample_rate=16000):
    """Core logic: Noise reduction"""
    return nr.reduce_noise(y=waveform_np, sr=sample_rate, prop_decrease=1.0)

def _normalize(waveform_np):
    """Core logic: Normalize to [-1, 1]"""
    max_val = np.max(np.abs(waveform_np))
    if max_val > 0:
        return waveform_np / max_val
    return waveform_np

def _segment(waveform_np, target_length):
    """Core logic: Pad or Slice"""
    current_length = waveform_np.shape[0]
    
    if current_length > target_length:
        return waveform_np[:target_length]
    elif current_length < target_length:
        pad_width = target_length - current_length
        return np.pad(waveform_np, (0, pad_width), mode='constant')
    
    return waveform_np


def _pitch_shift(waveform_np, sample_rate, n_steps=2):
    # Menggeser nada suara (augmentasi)
    return librosa.effects.pitch_shift(waveform_np, sr=sample_rate, n_steps=n_steps)


# ==========================================
#         WRAPPERS (Framework Specific)
# ==========================================

def load_audio(file_name, target_sr, backend='tf'):
    """
    Args:
        file_name: path audio
        backend: 'tf' (TensorFlow) atau 'pt' (PyTorch) atau 'np' (Numpy)
    """
    # 1. Jalankan Core Logic
    waveform_np = _load_audio(file_name, target_sr)
    
    # 2. Konversi sesuai backend
    if backend == 'tf':
        return tf.convert_to_tensor(waveform_np, dtype=tf.float32)
    elif backend == 'pt':
        return torch.from_numpy(waveform_np).float()
    else:
        return waveform_np

def silence_trimming(waveform, backend='tf', top_db=20):
    # Pastikan input jadi numpy dulu
    if backend == 'tf' and isinstance(waveform, tf.Tensor):
        wav_np = waveform.numpy()
    elif backend == 'pt' and isinstance(waveform, torch.Tensor):
        wav_np = waveform.numpy()
    else:
        wav_np = waveform
        
    trimmed_np = _trim_audio(wav_np, top_db)
    
    if backend == 'tf':
        return tf.convert_to_tensor(trimmed_np, dtype=tf.float32)
    elif backend == 'pt':
        return torch.from_numpy(trimmed_np).float()
    return trimmed_np

def reduce_noise(waveform, backend='tf', sample_rate=16000):
    if backend == 'tf' and isinstance(waveform, tf.Tensor):
        wav_np = waveform.numpy()
    elif backend == 'pt' and isinstance(waveform, torch.Tensor):
        wav_np = waveform.numpy()
    else:
        wav_np = waveform

    reduced_noise_np = _reduce_noise(wav_np, sample_rate=sample_rate)

    if backend == 'tf':
        return tf.convert_to_tensor(reduced_noise_np, dtype=tf.float32)
    elif backend == 'pt':
        return torch.from_numpy(reduced_noise_np).float()
    return reduced_noise_np

def normalize_audio(waveform, backend='tf'):
    if backend == 'tf' and isinstance(waveform, tf.Tensor):
        wav_np = waveform.numpy()
    elif backend == 'pt' and isinstance(waveform, torch.Tensor):
        wav_np = waveform.numpy()
    else:
        wav_np = waveform

    normalized_np = _normalize(wav_np)

    if backend == 'tf':
        return tf.convert_to_tensor(normalized_np, dtype=tf.float32)
    elif backend == 'pt':
        return torch.from_numpy(normalized_np).float()
    return normalized_np

def segment_audio(waveform, target_length, backend='tf'):
    if backend == 'tf' and isinstance(waveform, tf.Tensor):
        wav_np = waveform.numpy()
    elif backend == 'pt' and isinstance(waveform, torch.Tensor):
        wav_np = waveform.numpy()
    else:
        wav_np = waveform

    segmented_np = _segment(wav_np, target_length)

    if backend == 'tf':
        return tf.convert_to_tensor(segmented_np, dtype=tf.float32)
    elif backend == 'pt':
        return torch.from_numpy(segmented_np).float()
    return segmented_np


def apply_pitch_shift(waveform, backend='tf', sample_rate=16000, n_steps=2):
    if backend == 'tf' and isinstance(waveform, tf.Tensor):
        wav_np = waveform.numpy()
    elif backend == 'pt' and isinstance(waveform, torch.Tensor):
        wav_np = waveform.numpy()
    else:
        wav_np = waveform

    shifted_np = _pitch_shift(wav_np, sample_rate, n_steps)

    if backend == 'tf':
        return tf.convert_to_tensor(shifted_np, dtype=tf.float32)
    elif backend == 'pt':
        return torch.from_numpy(shifted_np).float()
    return shifted_np


import os
import numpy as np
import librosa
import random

def load_noise_files(noise_folder_path, sample_rate=16000):
    """
    Memuat semua file .wav dari folder background noise ke dalam list/dictionary.
    Dijalankan HANYA SEKALI di awal.
    """
    noise_dict = {}
    
    # Cek apakah folder ada
    if not os.path.exists(noise_folder_path):
        print(f"Error: Folder {noise_folder_path} tidak ditemukan.")
        return noise_dict

    files = [f for f in os.listdir(noise_folder_path) if f.endswith('.wav')]
    
    print(f"Memuat {len(files)} file noise...")
    
    for i, filename in enumerate(files):
        path = os.path.join(noise_folder_path, filename)
        try:
            # Load audio, pastikan sample rate sama dengan input audio utama Anda
            audio, _ = librosa.load(path, sr=sample_rate)
            # Simpan dengan key index atau nama file (disini kita pakai index agar mudah dirandom)
            noise_dict[i] = audio 
            print(f"Loaded: {filename}")
        except Exception as e:
            print(f"Gagal memuat {filename}: {e}")
            
    return noise_dict

def add_background_noise(data, noise_dict, noise_reduction=0.5):
    '''
    data: numpy array audio input (1D array)
    noise_dict: dictionary berisi numpy array noise (hasil dari load_noise_files)
    noise_reduction: 0.0 (sangat bising) sampai 1.0 (hening/tidak ada noise tambahan)
    '''
    
    # Jika noise_dict kosong atau noise_reduction 1 (100% reduction), kembalikan data asli
    if not noise_dict or noise_reduction >= 1.0:
        return data

    # 1. Pilih satu noise secara acak dari dictionary
    noise_id = random.choice(list(noise_dict.keys()))
    noise_data = noise_dict[noise_id]
    
    target_len = len(data)
    noise_len = len(noise_data)
    
    # 2. Random Cropping (Potong noise agar durasinya sama dengan input data)
    if noise_len > target_len:
        # Jika noise lebih panjang dari data, ambil potongan acak
        start_idx = np.random.randint(0, noise_len - target_len)
        noise_segment = noise_data[start_idx : start_idx + target_len]
    else:
        # Jika noise lebih pendek (jarang terjadi di dataset ini, tapi buat jaga-jaga)
        # Kita ulang (tile) noisenya sampai cukup panjang
        repeats = int(np.ceil(target_len / noise_len))
        noise_segment = np.tile(noise_data, repeats)[:target_len]
        
    # 3. Mixing
    # Konsep: Input Audio + (Noise * Volume Factor)
    # (1 - noise_reduction) berarti: jika reduction 0.8, maka volume noise cuma 0.2 (20%)
    noise_vol = 1.0 - noise_reduction
    data_with_noise = data + (noise_vol * noise_segment)
    
    # Opsional: Clipping agar tidak melebihi range audio float standar (-1.0 sampai 1.0)
    # data_with_noise = np.clip(data_with_noise, -1.0, 1.0)
    
    return data_with_noise.astype(np.float32)