import os
import json
import torch
import torchaudio
from tqdm import tqdm

# Import fungsi VAD dari file Anda (pastikan namanya sudah apply_vad.py)
from apply_vad import apply_silero_vad
from silero_vad import load_silero_vad

# ==========================================
# 1. PENGATURAN LOGISTIK (PATHS)
# ==========================================
RAW_AUDIO_DIR    = "data/raw/audio"
RAW_JSONL_PATH   = "data/processed/train_split.jsonl"

CLEAN_OUTPUT_DIR = "data/interim/mfa_corpus_train"
CLEAN_JSONL_PATH = "data/interim/clean_train_transcripts.jsonl"

os.makedirs(CLEAN_OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. FUNGSI PEMOTONG AUDIO
# ==========================================
def crop_and_concat_audio(audio_path, timestamps, target_sr=16000):
    """
    Membaca audio mentah, memotong bagian yang ada suaranya berdasarkan timestamps,
    dan menggabungkannya kembali menjadi satu audio yang padat (tanpa jeda hening).
    """
    try:
        # Load audio mentah
        wav, sr = torchaudio.load(audio_path)
        
        # Penyeragaman Otomatis: Ubah ke Mono jika Stereo
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
            
        # Penyeragaman Otomatis: Resample ke 16kHz jika berbeda
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)
            sr = target_sr
            
        # Jika VAD tidak mendeteksi suara sama sekali, kembalikan None
        if not timestamps:
            return None, sr

        # Gunting audio berdasarkan stempel waktu
        speech_segments = []
        for t in timestamps:
            start_sample = int(t['start'] * sr)
            end_sample = int(t['end'] * sr)
            speech_segments.append(wav[:, start_sample:end_sample])
            
        # Gabungkan (lem) semua potongan suara menjadi satu
        clean_wav = torch.cat(speech_segments, dim=1)
        return clean_wav, sr

    except Exception as e:
        print(f"\n[ERROR] Gagal memotong {audio_path}: {e}")
        return None, None

# ==========================================
# 3. MAIN PIPELINE (PABRIK BERJALAN)
# ==========================================
def main():
    print("🚀 Memulai Pabrik Pembersihan Data (VAD & MFA Prep)...")
    
    print("🤖 Memuat Model Silero VAD ke memori...")
    vad_model = load_silero_vad()
    
    with open(RAW_JSONL_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    sukses = 0
    gagal = 0
    clean_manifest = []

    print(f"📦 Ditemukan {len(lines)} file untuk diproses. Memulai pemotongan...")
    for line in tqdm(lines, desc="Processing Audio"):
        data = json.loads(line)
        
        # === PERBAIKAN ATRIBUT JSON ===
        file_id = data['utterance_id']
        transcript_text = data['orthographic_text']
        # ==============================
        
        original_audio_path = os.path.join(RAW_AUDIO_DIR, f"{file_id}.flac")
        
        # Langkah A: Pindai audio dengan VAD
        timestamps = apply_silero_vad(original_audio_path, vad_model)
        
        # Langkah B: Gunting dan Padatkan
        clean_wav, sr = crop_and_concat_audio(original_audio_path, timestamps)
        
        if clean_wav is not None:
            # Tentukan nama file baru
            new_audio_name = f"{file_id}_clean.flac"
            new_text_name = f"{file_id}_clean.txt"
            
            new_audio_path = os.path.join(CLEAN_OUTPUT_DIR, new_audio_name)
            new_text_path = os.path.join(CLEAN_OUTPUT_DIR, new_text_name)
            
            # Langkah C: Simpan Audio Bersih
            torchaudio.save(new_audio_path, clean_wav, sr)
            
            # Langkah D (TAKTIK MFA): Buat file .txt transkrip berdampingan dengan audio
            with open(new_text_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(transcript_text)
                
            # Langkah E: Catat di buku daftar (JSONL) baru
            data['audio_path'] = new_audio_path 
            clean_manifest.append(data)
            sukses += 1
        else:
            gagal += 1
            
        # ==========================================
        # TAKTIK BUMI HANGUS (PENGHEMATAN DISK LOKAL)
        # Hapus file mentah aslinya terlepas dari sukses atau gagalnya VAD
        # ==========================================
        # if os.path.exists(original_audio_path):
        #     os.remove(original_audio_path)
            
    # Simpan JSONL baru yang sudah diperbarui
    with open(CLEAN_JSONL_PATH, 'w', encoding='utf-8') as f:
        for item in clean_manifest:
            f.write(json.dumps(item) + '\n')
            
    print("\n✅ PABRIK SELESAI BEKERJA!")
    print(f"📊 Laporan: {sukses} file berhasil dibersihkan, {gagal} file diabaikan/gagal")
    print(f"📁 Folder MFA Corpus Anda siap di: {CLEAN_OUTPUT_DIR}")

if __name__ == "__main__":
    main()