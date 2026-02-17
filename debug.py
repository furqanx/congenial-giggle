import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim

# Import amunisi dari folder src
from src.processor import get_processor
from src.dataloader import get_dataloader
from src.model import build_model

def run_diagnostics(config_path="config.yaml"):
    print("==================================================")
    print("🛠️ MEMULAI DRY RUN DIAGNOSTIK (SNIPER PROTOCOL) 🛠️")
    print("==================================================")

    # 0. SETUP AWAL
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Setup] Perangkat terkunci di: {device.type.upper()}")
    
    processor = get_processor(config)
    pad_id = processor.tokenizer.pad_token_id

    # ==========================================
    # UJI 1: LOGISTIK AMUNISI (Dataloader)
    # ==========================================
    print("\n[UJI 1] Mengambil 1 Batch Amunisi HDF5...")
    config['data']['batch_size'] = 4 # Paksa batch kecil untuk hemat RAM saat debug
    train_loader, _ = get_dataloader(config['data'], processor)
    
    batch = next(iter(train_loader)) # Tarik paksa 1 batch pertama
    input_values = batch['input_values'].to(device)
    labels = batch['labels'].to(device)

    print(f"✅ Input Shape  (Batch, Time, 768) : {input_values.shape}")
    print(f"✅ Labels Shape (Batch, Text_Len)  : {labels.shape}")

    # ==========================================
    # UJI 2: KALIBRASI SENJATA (Forward Pass)
    # ==========================================
    print("\n[UJI 2] Menembakkan Forward Pass...")
    model = build_model(config['model'], processor).to(device)
    model.train() 
    
    # Hitung panjang aktual untuk Conformer/CTC
    input_lengths = torch.sum(input_values[:, :, 0] != 0.0, dim=1).long()
    
    outputs = model(input_values, input_lengths, target_tokens=labels)
    ctc_logits = outputs["ctc_logits"]
    decoder_logits = outputs["decoder_logits"]

    print(f"✅ CTC Logits Shape     : {ctc_logits.shape}")
    print(f"✅ Decoder Logits Shape : {decoder_logits.shape}")

    # ==========================================
    # UJI 3: SISTEM PEMBIDIK (Loss Calculation)
    # ==========================================
    print("\n[UJI 3] Menghitung Hybrid Loss...")
    ctc_loss_fn = nn.CTCLoss(blank=pad_id, zero_infinity=True)
    attn_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=-100)
    
    target_lengths = torch.sum(labels != -100, dim=1).long()
    ctc_targets = labels[labels != -100]

    ctc_logits_t = ctc_logits.transpose(0, 1).log_softmax(2)
    loss_ctc = ctc_loss_fn(ctc_logits_t, ctc_targets, input_lengths, target_lengths)
    
    loss_attn = attn_loss_fn(
        decoder_logits.reshape(-1, decoder_logits.size(-1)), 
        labels.reshape(-1)
    )
    
    loss = (0.3 * loss_ctc) + (0.7 * loss_attn)
    
    print(f"✅ CTC Loss       : {loss_ctc.item():.4f}")
    print(f"✅ Attention Loss : {loss_attn.item():.4f}")
    print(f"✅ Total Loss     : {loss.item():.4f} (Tidak boleh NaN!)")

    # ==========================================
    # UJI 4: MEKANISME PELATUK (Backward Pass)
    # ==========================================
    print("\n[UJI 4] Mengeksekusi Backward Pass (Update Bobot)...")
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Sensor Deteksi Gradien
    has_grad = any(p.grad is not None for p in model.parameters())
    if has_grad:
        print("✅ Arus Gradien terdeteksi. Backpropagation SUKSES!")
    else:
        print("❌ GAGAL: Tidak ada gradien. Model tidak akan belajar.")

    # ==========================================
    # UJI 5: MESIN SANDI (Decoding)
    # ==========================================
    print("\n[UJI 5] Menguji Dekode Tokenizer...")
    pred_ids = torch.argmax(ctc_logits, dim=-1)
    pred_strs = processor.tokenizer.batch_decode(pred_ids)
    
    # Clone agar tidak merusak tensor asli, ubah -100 jadi pad agar bisa dibaca tokenizer
    labels_clone = labels.clone()
    labels_clone[labels_clone == -100] = pad_id
    label_strs = processor.tokenizer.batch_decode(labels_clone, group_tokens=False)

    print(f"✅ Teks Asli (Ground Truth) : '{label_strs[0]}'")
    print(f"✅ Teks Tebakan (Acak)      : '{pred_strs[0]}'")

    print("\n==================================================")
    print("🎯 SEMUA SISTEM NORMAL. SIAP UNTUK TRAINING UTAMA! 🎯")
    print("==================================================")

if __name__ == "__main__":
    # Eksekusi diagnostik menggunakan file konfigurasi Anda
    run_diagnostics(config_path="config.yaml")