# 🎯 Child Speech ASR: Hybrid CTC/Attention Architecture

Repositori ini berisi *pipeline* *State-of-the-Art* (SOTA) Automatic Speech Recognition (ASR) yang dirancang khusus untuk mengenali suara anak-anak yang penuh tantangan (noise, artikulasi tidak jelas). 

Alih-alih melatih model End-to-End dari audio mentah, arsitektur ini menggunakan pendekatan **"Offline SSL Extraction + Lightweight Hybrid Downstream"**:
1. **Offline Extraction**: Mengekstrak fitur akustik menggunakan model SSL raksasa (WavLM) menjadi matriks 768-dimensi yang disimpan dalam brankas efisien `.h5` (HDF5).
2. **Native PyTorch Downstream**: Menggunakan model ringan (BiLSTM/Conformer Encoder + Transformer Decoder) untuk melakukan *training* dengan Hybrid Loss (CTC + Attention dengan Label Smoothing) yang sangat hemat VRAM.

## 📁 Struktur Markas (Repository Tree)

```text
.
├── data/
│   ├── raw/                      # Tempat meletakkan audio (.flac) dan manifest mentah
│   └── processed/                # Hasil split (train/val/test.jsonl) & brankas .h5
├── logs/                         # Hasil eksperimen, config backup, dan checkpoint model
├── src/
│   ├── utils/
│   │   ├── metrics.py            # Kalkulasi WER/CER dengan dekontaminasi teks (jiwer)
│   │   └── split_data.py         # Skrip pembedahan dataset (Train/Val/Test)
│   ├── dataloader.py             # Dataloader berbasis HDF5 dengan Lazy Loading
│   ├── model.py                  # Arsitektur PyTorch Native (BiLSTM/Conformer + Decoder)
│   ├── processor.py              # Custom Tokenizer dengan token spesial (<s>, </s>, [PAD])
│   └── trainer.py                # Loop training dengan AMP (FP16) & Hybrid Loss
├── config.yaml                   # Pusat komando (Hyperparameters, Paths, Arsitektur)
├── extract_embeddings.py         # Eksekutor ekstraksi WavLM -> HDF5
├── debug.py                      # Skrip Dry-Run (Diagnostic test 1-batch)
└── main.py                       # Skrip eksekusi utama (Training)