import os
import torch
import torchaudio
import numpy as np
import h5py  
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel, WhisperModel

# ==========================================
# KONFIGURASI ARSENAL (4 MODEL PILIHAN)
# ==========================================
MODELS_CONFIG = {
    "wavlm": "microsoft/wavlm-base-plus",
    "hubert": "facebook/hubert-base-ls960",
    "wav2vec2": "facebook/wav2vec2-base",
    "whisper": "openai/whisper-base.en" 
}

def extract_and_save_embeddings(audio_dir, output_base_dir, model_key="wavlm"):
    """
    Mengekstrak fitur vektor (embeddings) dari audio menggunakan model SOTA.
    Disimpan secara terpusat dan terkompresi di dalam SATU file .h5 (HDF5).
    """
    if model_key not in MODELS_CONFIG:
        raise ValueError(f"Model '{model_key}' tidak dikenali. Pilih dari: {list(MODELS_CONFIG.keys())}")
        
    model_id = MODELS_CONFIG[model_key]
    print(f"\n🎯 MENGUNCI TARGET MODEL: {model_key.upper()} ({model_id})")
    
    # 1. Persiapan Perangkat
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Perangkat komputasi: {device.type.upper()}")

    # 2. Load Extractor dan Model
    print("⏳ Memuat model ke memori...")
    extractor = AutoFeatureExtractor.from_pretrained(model_id)
    
    if model_key == "whisper":
        model = WhisperModel.from_pretrained(model_id).get_encoder().to(device)
    else:
        model = AutoModel.from_pretrained(model_id).to(device)
        
    model.eval()

    # 3. Persiapan Brankas HDF5 (.h5)
    os.makedirs(output_base_dir, exist_ok=True)
    h5_filename = f"{model_key}_embeddings.h5"
    h5_filepath = os.path.join(output_base_dir, h5_filename)
    print(f"📁 Target brankas penyimpanan: {h5_filepath}")

    # 4. Mencari File Audio
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith(".flac")]
    if not audio_files:
        print("⚠️ Tidak ada file .flac ditemukan di direktori sumber!")
        return

    print(f"🚀 Memulai ekstraksi untuk {len(audio_files)} file...")

    # 5. Eksekusi Ekstraksi (Membuka file HDF5 dalam mode 'Append')
    # Mode 'a' memastikan jika script mati, ia akan melanjutkannya, bukan menimpanya dari awal.
    with h5py.File(h5_filepath, 'a') as h5f, torch.no_grad(): 
        
        for filename in tqdm(audio_files, desc=f"Ekstraksi {model_key.upper()}"):
            audio_path = os.path.join(audio_dir, filename)
            
            # Gunakan nama file (tanpa .flac) sebagai kunci indeks di HDF5
            audio_id = os.path.splitext(filename)[0] 

            # Lewati jika ID ini sudah tersimpan di dalam brankas (Fitur Anti-Crash)
            if audio_id in h5f:
                continue

            try:
                # Load dan preprocess audio
                waveform, sample_rate = torchaudio.load(audio_path)
                if sample_rate != 16000:
                    waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=16000)
                
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # Ekstraksi menggunakan model
                if model_key == "whisper":
                    inputs = extractor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
                    input_features = inputs.input_features.to(device)
                    outputs = model(input_features)
                else:
                    inputs = extractor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
                    input_values = inputs.input_values.to(device)
                    outputs = model(input_values)

                # Ambil matriks 2D (Waktu x 768), dan PAKSA menjadi float16
                embeddings = outputs.last_hidden_state.squeeze(0).cpu().numpy()
                # embeddings = outputs.last_hidden_state.squeeze(0).cpu().numpy().astype(np.float16)

                # Simpan ke dalam HDF5 sebagai dataset baru dengan kompresi tingkat 4
                h5f.create_dataset(
                    name=audio_id,
                    data=embeddings,
                    dtype=np.float16,
                    compression="gzip",
                    compression_opts=4 
                )

            except Exception as e:
                print(f"\n❌ Gagal memproses {filename}: {e}")

    print("==================================================")
    print(f"✅ EKSTRAKSI {model_key.upper()} SELESAI!")
    print(f"📦 Hasil tersimpan rapat di: {h5_filepath}")
    print("==================================================")

if __name__ == "__main__":
    DIR_AUDIO = "data/raw/audio"
    DIR_OUTPUT = "data/processed/embeddings"
    
    # Pilih senjatanya: "wavlm", "hubert", "wav2vec2", atau "whisper"
    MODEL_YANG_DIGUNAKAN = "wavlm" 
    
    extract_and_save_embeddings(
        audio_dir=DIR_AUDIO, 
        output_base_dir=DIR_OUTPUT, 
        model_key=MODEL_YANG_DIGUNAKAN
    )