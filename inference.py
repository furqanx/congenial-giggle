import os
import argparse
import yaml
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from src.model import build_model

class InferenceEngine:
    def __init__(self, config_path, checkpoint_path, device=None):
        """
        Inisialisasi Inference Engine.
        Memuat model dan bobot (checkpoint) ke memori.
        """
        # 1. Setup Device
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Inference] Using device: {self.device}")

        # 2. Load Config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # 3. Build Model Structure
        print(f"[Inference] Building model: {self.config['model']['name']}...")
        self.model = build_model(self.config['model'])
        self.model = self.model.to(self.device)

        # 4. Load Weights (Checkpoint)
        print(f"[Inference] Loading checkpoint from: {checkpoint_path}")
        self.load_checkpoint(checkpoint_path)
        
        # 5. Set to Eval Mode (Matikan Dropout/BatchNorm update)
        self.model.eval()

    def load_checkpoint(self, path):
        """Menangani loading bobot, baik format dictionary lengkap maupun state_dict saja."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at {path}")
            
        checkpoint = torch.load(path, map_location=self.device)
        
        # Cek apakah checkpoint menyimpan optimizer state dll (biasanya training loop menyimpan dict)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint # Kasus jika cuma save model weights saja

        # Load ke model
        try:
            self.model.load_state_dict(state_dict)
        except Exception as e:
            print(f"[Warning] Strict loading failed, trying loose loading... Error: {e}")
            # Kadang nama layer ada prefix 'module.' jika dilatih pakai DataParallel
            # Kita bisa bersihkan key-nya atau load dengan strict=False
            self.model.load_state_dict(state_dict, strict=False)

    def preprocess(self, audio_path, max_sec=3.0):
        """
        Memuat audio, resample ke 16k, dan memotong/padding.
        Output: Tensor (1, Time)
        """
        # Load audio (format tensor: Channel, Time)
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample jika bukan 16000Hz
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)

        # Ubah ke Mono jika Stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Opsional: Fixed Length (Misal untuk konsistensi batch, tapi untuk inference single file bebas)
        # Di sini kita biarkan variable length atau potong jika terlalu panjang untuk hemat memori
        # max_len = int(16000 * max_sec)
        # if waveform.shape[1] > max_len:
        #     waveform = waveform[:, :max_len]
        
        # Tambahkan dimensi Batch: (1, 1, Time) -> (1, Time) karena model kita terima (Batch, Time)
        # Model ResNetSE/RawNet kita mengharapkan input (Batch, Time)
        waveform = waveform.squeeze(0) # Jadi (Time, )
        waveform = waveform.unsqueeze(0) # Jadi (1, Time)
        
        return waveform.to(self.device)

    def get_embedding(self, audio_path):
        """
        Forward pass untuk mendapatkan vektor fitur.
        """
        input_tensor = self.preprocess(audio_path)
        
        with torch.no_grad():
            embedding = self.model(input_tensor)
            # L2 Normalize (Standard untuk Speaker Verification)
            # Agar Cosine Similarity valid
            embedding = F.normalize(embedding, p=2, dim=1)
            
        return embedding.cpu().numpy()[0] # Return sebagai numpy array 1D

    def compute_similarity(self, embed1, embed2):
        """
        Menghitung Cosine Similarity antara dua embedding.
        Range: -1 (Beda total) s/d 1 (Sama persis).
        """
        # Karena sudah di-normalize L2 di get_embedding,
        # Cosine Similiarity = Dot Product
        score = np.dot(embed1, embed2)
        return score

# --- CLI HANDLING ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speaker Verification Inference")
    parser.add_argument('--config', type=str, required=True, help='Path to config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to .pth model file')
    parser.add_argument('--mode', type=str, default='verify', choices=['embed', 'verify'], help='Mode: "embed" (single file) or "verify" (compare two files)')
    parser.add_argument('--audio1', type=str, required=True, help='Path to first audio file')
    parser.add_argument('--audio2', type=str, help='Path to second audio file (required for verify mode)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Similarity threshold for decision')

    args = parser.parse_args()

    # 1. Init Engine
    engine = InferenceEngine(args.config, args.checkpoint)

    # 2. Execute based on mode
    if args.mode == 'embed':
        emb = engine.get_embedding(args.audio1)
        print(f"\n[Result] Embedding Generated. Shape: {emb.shape}")
        print(f"First 10 values: {emb[:10]}")

    elif args.mode == 'verify':
        if not args.audio2:
            raise ValueError("Mode 'verify' requires --audio2 argument.")
        
        print(f"\nProcessing Audio 1: {args.audio1}")
        emb1 = engine.get_embedding(args.audio1)
        
        print(f"Processing Audio 2: {args.audio2}")
        emb2 = engine.get_embedding(args.audio2)
        
        score = engine.compute_similarity(emb1, emb2)
        
        is_same = score > args.threshold
        print("\n" + "="*30)
        print(f"SIMILARITY SCORE: {score:.4f}")
        print(f"THRESHOLD       : {args.threshold}")
        print(f"PREDICTION      : {'SAME SPEAKER' if is_same else 'DIFFERENT SPEAKER'}")
        print("="*30)