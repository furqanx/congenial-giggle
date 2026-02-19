import os
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorCTCWithPadding:
    processor: Any
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Pisahkan input (audio) dan label (teks)
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # 1. Padding Audio (Menggunakan FeatureExtractor bawaan Processor)
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # 2. Padding Teks Label (Menggunakan Tokenizer bawaan Processor)
        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # Ganti padding teks dengan -100 agar diabaikan PyTorch saat menghitung CTC Loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels
        return batch

class ASRDataset(Dataset):
    def __init__(self, data, processor, target_sr=16000):
        self.data = data
        self.processor = processor
        self.target_sr = target_sr # Standar WavLM adalah 16000 Hz

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Ekstrak data menggunakan nama kolom baru dari JSONL kita
        audio_path = row['audio_filepath']
        transcript = row['text']

        # 1. Load Raw Audio menggunakan torchaudio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Ubah ke mono jika audionya stereo (2 channel)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Resample audio jika frekuensinya tidak 16kHz
        if sample_rate != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sr)
            waveform = resampler(waveform)

        # Buang dimensi channel menjadi 1D array agar bisa ditelan Processor HuggingFace
        waveform = waveform.squeeze(0).numpy()

        # 2. Ekstrak Fitur Audio (Normalisasi zero mean unit variance)
        # Processor akan menormalisasi volume suara anak-anak yang kadang pelan/keras
        input_values = self.processor(waveform, sampling_rate=self.target_sr).input_values[0]

        # 3. Tokenisasi Teks
        labels = self.processor.tokenizer(transcript).input_ids

        return {
            "input_values": input_values,
            "labels": labels
        }

def filter_data(df, min_duration=0.5, max_duration=15.0):
    # Menyaring audio. Nama kolom disesuaikan dengan output jsonl kita ('duration')
    if 'duration' in df.columns:
        return df[(df['duration'] >= min_duration) & (df['duration'] <= max_duration)]
    return df

def get_dataloader(config, processor):
    print(f"[DataLoader] Membaca manifest: {config['train_manifest']}")
    
    train_df = pd.read_json(config['train_manifest'], lines=True)
    val_df = pd.read_json(config['val_manifest'], lines=True)
    
    # Filter Data untuk mencegah OOM (Out of Memory)
    max_dur = config.get('max_duration', 15.0)
    train_df = filter_data(train_df, max_duration=max_dur)
    val_df = filter_data(val_df, max_duration=20.0)

    target_sr = config.get('sample_rate', 16000)

    # Setup Dataset (Sekarang murni menggunakan Audio + Teks)
    train_ds = ASRDataset(data=train_df, processor=processor, target_sr=target_sr)
    val_ds = ASRDataset(data=val_df, processor=processor, target_sr=target_sr)

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