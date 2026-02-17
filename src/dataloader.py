import os
import torch
import pandas as pd
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorCTCWithPadding:
    processor: Any
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        
        # 1. Ekstraksi Vektor dan Label
        # input_values sekarang adalah Tensor 2D dengan bentuk (Waktu, 768)
        input_features = [feature["input_values"] for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # 2. Padding Vektor Audio (Manual PyTorch)
        # Menyamakan panjang waktu semua vektor dalam satu batch dengan padding angka 0.0
        # Hasilnya: Tensor 3D berdimensi (Batch, Max_Time, 768)
        batch_input_values = pad_sequence(input_features, batch_first=True, padding_value=0.0)

        # 3. Padding Teks Label (Tetap menggunakan Tokenizer)
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # Ganti padding teks dengan -100 agar diabaikan saat menghitung Loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        return {
            "input_values": batch_input_values, # Vektor cerdas (Batch, Waktu, 768)
            "labels": labels                    # Teks target (Batch, Teks)
        }

class ASRDataset(Dataset):
    def __init__(self, data, processor, h5_path):
        self.data = data
        self.processor = processor
        self.h5_path = h5_path
        
        # PENTING: Jangan buka file h5py di sini jika menggunakan num_workers > 0
        # PyTorch Multiprocessing bisa crash jika file dibuka di __init__
        self.h5_file = None 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Buka brankas secara "Lazy" saat data benar-benar dipanggil
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')

        row = self.data.iloc[idx]
        
        # Ekstrak ID (Contoh: "audio/U_000.flac" -> "U_000")
        audio_filename = os.path.basename(row['audio_path'])
        audio_id = os.path.splitext(audio_filename)[0]
        
        transcript = row['orthographic_text'] 

        # 1. Ambil matriks float16 dari brankas HDF5
        embedding_fp16 = self.h5_file[audio_id][:]
        
        # 2. Konversi kembali ke float32 (Syarat wajib untuk proses Training PyTorch)
        embedding_tensor = torch.tensor(embedding_fp16, dtype=torch.float32)

        # 3. Tokenisasi Teks
        labels = self.processor.tokenizer(transcript).input_ids

        return {
            "input_values": embedding_tensor,
            "labels": labels
        }

def filter_data(df, min_duration=1.0, max_duration=15.0):
    # Menyaring audio yang terlalu panjang atau terlalu pendek
    return df[(df['audio_duration_sec'] >= min_duration) & (df['audio_duration_sec'] <= max_duration)]

def get_dataloader(config, processor):
    print(f"[DataLoader] Membaca manifest: {config['train_manifest']}")
    
    train_df = pd.read_json(config['train_manifest'], lines=True)
    val_df = pd.read_json(config['val_manifest'], lines=True)
    
    # Filter Data (Pastikan kolom di JSONL Anda benar-benar bernama 'audio_duration_sec')
    max_dur = config.get('max_duration', 15.0)
    train_df = filter_data(train_df, max_duration=max_dur)
    val_df = filter_data(val_df, max_duration=20.0)

    # --- PERBAIKAN TAKTIS DI SINI ---
    # Memisahkan jalur brankas HDF5 untuk Train dan Val
    h5_train_path = config.get('h5_train_path', 'data/processed/embeddings/train_wavlm.h5')
    h5_val_path = config.get('h5_val_path', 'data/processed/embeddings/val_wavlm.h5')

    # Setup Dataset
    train_ds = ASRDataset(
        data=train_df, 
        processor=processor, 
        h5_path=h5_train_path
    )
    
    val_ds = ASRDataset(
        data=val_df, 
        processor=processor, 
        h5_path=h5_val_path # Sekarang membaca dari brankas yang benar
    )

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 2),
        collate_fn=data_collator, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 2),
        collate_fn=data_collator,
        pin_memory=True
    )
    
    return train_loader, val_loader