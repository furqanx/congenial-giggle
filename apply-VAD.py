import torch
import whisper
from pyannote.audio import Pipeline
from silero_vad import read_audio, get_speech_timestamps, load_silero_vad

# ==========================================
# SILERO VAD
# ==========================================

def apply_silero_vad(
        audio_path, 
        vad_model, 
        threshold=0.3, 
        min_silence_duration_ms=500, 
        min_speech_duration_ms=250
    ):
    """
    Mengekstrak timestamp aktivitas suara (VAD) menggunakan Silero VAD.
    Dioptimalkan untuk suara anak-anak dengan threshold rendah dan toleransi jeda tinggi.

    Parameter:
    - audio_path (str): Jalur lengkap menuju file audio (.wav, .flac, dll).
    - vad_model: Objek model Silero VAD yang sudah di-load sebelumnya.
    - threshold (float): Batas keyakinan (0-1). Anak-anak butuh threshold lebih rendah (contoh: 0.3).
    - min_silence_duration_ms (int): Jeda minimal (ms) untuk memotong suara. Dinaikkan ke 500ms agar napas/jeda anak tidak memotong kata.
    - min_speech_duration_ms (int): Durasi minimal (ms) agar sesuatu dianggap suara manusia.

    Return:
    - list of dict: Berisi timestamp dalam detik, misal [{'start': 1.2, 'end': 3.5}, ...]
    """
    
    try:
        # 1. Membaca file audio (Silero secara otomatis mengubahnya menjadi tensor dan meresample ke 16kHz)
        wav = read_audio(audio_path)
        
        # 2. Mengekstrak timestamp
        speech_timestamps = get_speech_timestamps(
            wav,
            vad_model,
            threshold=threshold,
            min_silence_duration_ms=min_silence_duration_ms,
            min_speech_duration_ms=min_speech_duration_ms,
            return_seconds=True # Mengembalikan dalam format detik, bukan indeks sampel
        )
        
        return speech_timestamps

    except Exception as e:
        print(f"Error memproses {audio_path}: {e}")
        return []
    

# ==========================================
# PYANNOTE VAD
# ==========================================
def apply_pyannote_vad(audio_path, vad_pipeline):
    """
    Mengekstrak timestamp aktivitas suara (VAD) menggunakan Pyannote Audio.
    Sangat presisi untuk batas kata (word boundaries).

    Parameter:
    - audio_path (str): Jalur lengkap menuju file audio (.wav, .flac, dll).
    - vad_pipeline: Objek pipeline Pyannote yang sudah di-load sebelumnya.

    Return:
    - list of dict: Berisi timestamp dalam detik, konsisten dengan format output Silero.
                    Contoh: [{'start': 1.2, 'end': 3.5}, ...]
    """
    try:
        # Menjalankan pipeline inferensi pada file audio
        vad_results = vad_pipeline(audio_path)
        
        speech_timestamps = []
        
        # Pyannote mengembalikan objek 'Annotation'. 
        # Kita ekstrak segment-nya (start dan end dalam detik).
        for segment in vad_results.get_timeline().support():
            speech_timestamps.append({
                'start': round(segment.start, 3), # Dibulatkan 3 angka di belakang koma (milidetik)
                'end': round(segment.end, 3)
            })
            
        return speech_timestamps

    except Exception as e:
        print(f"Error memproses {audio_path} dengan Pyannote: {e}")
        return []
    
# ==========================================
# FUNGSI 3: WHISPER (SEBAGAI VAD / TIMESTAMPING)
# ==========================================
def apply_whisper_vad(audio_path, whisper_model):
    """
    Mengekstrak timestamp aktivitas suara menggunakan OpenAI Whisper.
    Whisper sangat cerdas mengenali suara berkonteks linguistik (termasuk anak-anak) 
    meskipun di tengah noise, karena ia juga mencoba mentranskripsinya di latar belakang.

    Parameter:
    - audio_path (str): Jalur lengkap menuju file audio (.wav, .mp3, dll).
    - whisper_model: Objek model Whisper yang sudah di-load sebelumnya.

    Return:
    - list of dict: Berisi timestamp dalam detik, konsisten dengan Silero dan Pyannote.
                    Contoh: [{'start': 1.2, 'end': 3.5}, ...]
    """
    # ==========================================
    # DAFTAR PILIHAN MODEL WHISPER (LOKAL/OPEN-SOURCE)
    # Sesuaikan dengan kapasitas VRAM GPU Anda:
    # 
    # 1. 'tiny'     (39 Juta Parameter)  -> VRAM ~1 GB. Sangat Direkomendasikan untuk VAD. Paling cepat.
    # 2. 'base'     (74 Juta Parameter)  -> VRAM ~1 GB. Cepat dan ringan.
    # 3. 'small'    (244 Juta Parameter) -> VRAM ~2 GB. Cukup akurat untuk menangkap gumaman.
    # 4. 'medium'   (769 Juta Parameter) -> VRAM ~5 GB. Agak lambat kalau memproses data ratusan jam.
    # 5. 'large-v3' (1.5 Miliar Parameter)-> VRAM ~10 GB. Sangat akurat tapi OVERKILL (terlalu berat) jika hanya untuk VAD.
    # 
    # *Tips: Tambahkan akhiran '.en' (contoh: 'tiny.en') jika dataset 100% bahasa Inggris agar lebih akurat.
    # ==========================================
    try:
        # Kita panggil fungsi transcribe. 
        # Secara otomatis Whisper memecah audio menjadi 'segments' yang memiliki timestamp.
        # fp16=False digunakan jika CPU, fp16=True jika menggunakan GPU (CUDA).
        use_fp16 = torch.cuda.is_available()
        
        result = whisper_model.transcribe(
            audio_path, 
            fp16=use_fp16, 
            word_timestamps=False # Kita set False karena kita butuh batas segmen/kalimat, bukan per kata
        )
        
        speech_timestamps = []
        
        # Ekstrak waktu mulai dan selesai dari setiap segmen suara yang terdeteksi
        for segment in result["segments"]:
            speech_timestamps.append({
                'start': round(segment['start'], 3),
                'end': round(segment['end'], 3)
            })
            
        return speech_timestamps

    except Exception as e:
        print(f"Error memproses {audio_path} dengan Whisper: {e}")
        return []
    
# ==========================================
# SIMULASI MAIN PIPELINE LENGKAP
# ==========================================
if __name__ == "__main__":
    file_uji = "sampel_suara_anak.wav" # File dummy untuk test
    
    # ------------------------------------------
    # TEST 3: WHISPER VAD
    # ------------------------------------------
    print("Loading Whisper model ('tiny')...")
    # Anda bisa mengganti 'tiny' dengan 'base', 'small', 'medium', 'tiny.en', dll.
    
    # Otomatis menggunakan GPU jika tersedia
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    whisper_model = whisper.load_model("tiny", device=device)
    
    # Panggil fungsi
    # print(f"Hasil Whisper: {apply_whisper_vad(file_uji, whisper_model)}")
    
    print("\nSemua model (Silero, Pyannote, Whisper) siap dieksekusi!")

# ==========================================
# CONTOH CARA PENGGUNAAN DI MAIN PIPELINE NANTI
# ==========================================
if __name__ == "__main__":
    
    # 1. Load model HANYA SATU KALI di awal main pipeline
    print("Loading Silero VAD model...")
    model = load_silero_vad()
    
    # 2. Loop melalui ribuan file Anda
    file_uji = "sampel_suara_anak.wav" # Ganti dengan file asli Anda saat mencoba
    
    # Panggil fungsi
    hasil_timestamps = apply_silero_vad(
        audio_path=file_uji,
        vad_model=model
    )
    
    print(f"Hasil VAD untuk {file_uji}:")
    print(hasil_timestamps) 
    # Output yang diharapkan: [{'start': 0.5, 'end': 2.1}, {'start': 3.0, 'end': 4.5}]

# ==========================================
# SIMULASI MAIN PIPELINE (CARA LOAD KEDUANYA)
# ==========================================
if __name__ == "__main__":
    file_uji = "sampel_suara_anak.wav" # File dummy untuk test
    
    # ------------------------------------------
    # TEST 1: SILERO VAD
    # ------------------------------------------
    print("Loading Silero VAD model...")
    silero_model = load_silero_vad()
    
    # print(f"Hasil Silero: {apply_silero_vad(file_uji, silero_model)}")
    
    # ------------------------------------------
    # TEST 2: PYANNOTE VAD
    # ------------------------------------------
    print("Loading Pyannote VAD pipeline...")
    
    # MASUKKAN TOKEN HUGGING FACE ANDA DI SINI
    HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" 
    
    try:
        # Load pipeline hanya SATU KALI sebelum masuk ke loop dataset
        pyannote_pipeline = Pipeline.from_pretrained(
            "pyannote/voice-activity-detection",
            use_auth_token=HF_TOKEN
        )
        
        # Pindahkan ke GPU jika tersedia agar inferensi super cepat
        if torch.cuda.is_available():
            pyannote_pipeline.to(torch.device("cuda"))
            print("Pyannote menggunakan GPU (CUDA).")
        else:
            print("Pyannote menggunakan CPU.")
            
        # Panggil fungsi
        # print(f"Hasil Pyannote: {apply_pyannote_vad(file_uji, pyannote_pipeline)}")
        
    except Exception as e:
        print(f"Gagal meload Pyannote: {e}")
        print("Pastikan token Anda benar dan Anda sudah Accept Conditions di website HF.")