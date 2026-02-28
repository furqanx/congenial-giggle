import os
import random
import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# Mengimpor senjata rahasia kita
from src.preprocessing import DynamicNoiseInjector

# ==========================================
# 1. DATA COLLATOR (Tukang Packing)
# ==========================================
@dataclass
class DataCollatorCTCWithPadding:
    processor: Any
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # labels = labels_batch["input_ids"].masked_fill(
        #     labels_batch.attention_mask.ne(1), -100
        # )

        labels = labels_batch["input_ids"]
        labels = labels.masked_fill(labels == self.processor.tokenizer.pad_token_id, -100)

        batch["labels"] = labels
        return batch

# ==========================================
# 2. BUCKET BATCH SAMPLER (Sang Penghemat Waktu)
# ==========================================
class BucketBatchSampler(Sampler):
    """
    Mengelompokkan audio dengan durasi yang mirip ke dalam satu batch.
    Mencegah GPU membuang waktu menghitung angka 0 (padding) untuk file pendek
    yang tidak sengaja disatukan dengan file sangat panjang.
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Ambil durasi dari setiap baris data
        self.ind_n_len = []
        for i in range(len(dataset)):
            self.ind_n_len.append((i, dataset.data.iloc[i]['duration']))
        
        # Urutkan berdasarkan durasi (pendek ke panjang)
        self.ind_n_len.sort(key=lambda x: x[1])
        
        # Potong-potong menjadi keranjang (bucket) sesuai batch_size
        self.batches = [self.ind_n_len[i:i + batch_size] for i in range(0, len(self.ind_n_len), batch_size)]
        
    def __iter__(self):
        # Acak urutan KERANJANG-nya agar model tidak selalu belajar dari yang pendek ke panjang tiap epoch
        random.shuffle(self.batches)
        for batch in self.batches:
            yield [idx for idx, _ in batch] # Kembalikan ID datanya saja
            
    def __len__(self):
        return len(self.batches)

# ==========================================
# 3. DATASET KUSTOM
# ==========================================
class ASRDataset(Dataset):
    def __init__(self, data, processor, target_sr=16000, augmentor=None):
        self.data = data
        self.processor = processor
        self.target_sr = target_sr 
        self.augmentor = augmentor # Komponen baru: Penyuntik Noise

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        audio_path = row['audio_filepath']
        transcript = row['text']

        waveform, sample_rate = torchaudio.load(audio_path)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        if sample_rate != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sr)
            waveform = resampler(waveform)

        waveform = waveform.squeeze(0).numpy()

        # --- INJEKSI NOISE (Hanya jika augmentor diaktifkan) ---
        if self.augmentor is not None:
            waveform = self.augmentor(waveform_np=waveform, sample_rate=self.target_sr)

        input_values = self.processor(waveform, sampling_rate=self.target_sr).input_values[0]
        labels = self.processor.tokenizer(transcript).input_ids

        return {
            "input_values": input_values,
            "labels": labels
        }

def filter_data(df, min_duration=0.5, max_duration=15.0):
    if 'duration' in df.columns:
        return df[(df['duration'] >= min_duration) & (df['duration'] <= max_duration)].reset_index(drop=True)
    return df

# ==========================================
# 4. GET DATALOADER (Perakit Akhir)
# ==========================================
def get_dataloader(config, processor):
    print(f"[DataLoader] Membaca manifest: {config['train_manifest']}")
    
    train_df = pd.read_json(config['train_manifest'], lines=True)
    val_df = pd.read_json(config['val_manifest'], lines=True)
    
    max_dur = config.get('max_duration', 15.0)
    train_df = filter_data(train_df, max_duration=max_dur)
    val_df = filter_data(val_df, max_duration=20.0)

    target_sr = config.get('sample_rate', 16000)

    # --- SETUP NOISE INJECTOR ---
    # Path folder noise diatur secara default, pastikan folder /content/data/raw/noise ada di Colab
    noise_dir = config.get('noise_dir', '/content/data/raw/noise')
    
    noise_injector = None
    if os.path.exists(noise_dir):
        # Probabilitas 50% data train kena noise
        noise_injector = DynamicNoiseInjector(noise_dir=noise_dir, p=0.5)
    else:
        print(f"⚠️ Peringatan: Folder noise '{noise_dir}' tidak ditemukan. Augmentasi dinonaktifkan.")

    # Setup Dataset (Noise HANYA untuk Data Train, Data Val harus tetap bersih/murni)
    train_ds = ASRDataset(data=train_df, processor=processor, target_sr=target_sr, augmentor=noise_injector)
    val_ds = ASRDataset(data=val_df, processor=processor, target_sr=target_sr, augmentor=None)

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    # Memasang BucketBatchSampler ke Train Loader
    train_sampler = BucketBatchSampler(train_ds, batch_size=config['batch_size'])

    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler, # Gantikan parameter batch_size dan shuffle
        num_workers=config.get('num_workers', 2),
        collate_fn=data_collator, 
        pin_memory=True
    )
    
    # Val Loader tidak perlu Bucket, cukup mode standar
    val_loader = DataLoader(
        val_ds,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 2),
        collate_fn=data_collator,
        pin_memory=True
    )
    
    return train_loader, val_loader