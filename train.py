import os
import sys
import yaml
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup

# --- Import Modul Custom Kita ---
from src.dataloader import get_dataloader
from src.processor import get_processor
from src.model import build_model
from src.trainer import ASRTrainer # Pastikan nama class di src/trainer.py adalah ASRTrainer

def set_seed(seed):
    """Mengatur seed agar hasil eksperimen bisa direproduksi (reproducible)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[Setup] Seed set to: {seed}")

def main(args):
    # ====================================================
    # 1. SETUP & CONFIGURATION
    # ====================================================
    print(f"[Main] Loading configuration from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Buat folder output eksperimen
    output_dir = os.path.join(config['experiment']['output_dir'], config['experiment']['project_name'])
    os.makedirs(output_dir, exist_ok=True)
    
    # Simpan config yang dipakai sebagai arsip
    with open(os.path.join(output_dir, "config_saved.yaml"), "w") as f:
        yaml.dump(config, f)

    set_seed(config['experiment']['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Main] Device: {device}")

    # ====================================================
    # 2. PREPARE PROCESSOR (VOCAB & TOKENIZER)
    # ====================================================
    processor = get_processor(config)
    
    # ====================================================
    # 3. DATA PIPELINE
    # ====================================================
    train_loader, val_loader = get_dataloader(config['data'], processor)
    print(f"[Main] Data loaded. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ====================================================
    # 4. BUILD MODEL
    # ====================================================
    model = build_model(config['model'], processor)
    model.to(device)

    # ====================================================
    # 5. OPTIMIZER & SCHEDULER
    # ====================================================
    print("[Main] Configuring Optimizer & Scheduler...")
    
    # Hyperparameters
    lr = float(config['train']['learning_rate'])
    weight_decay = float(config['train'].get('weight_decay', 0.005))
    epochs = int(config['train']['epochs'])
    accum_steps = int(config['train'].get('gradient_accumulation_steps', 1))

    # Optimizer: AdamW adalah standar untuk Transformer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=lr,
        weight_decay=weight_decay
    )

    num_update_steps_per_epoch = len(train_loader) // accum_steps
    max_train_steps = epochs * num_update_steps_per_epoch
    
    warmup_steps = int(0.1 * max_train_steps) 
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps
    )
    
    print(f"[Optimizer] Total Steps: {max_train_steps} | Warmup Steps: {warmup_steps}")

    # ====================================================
    # 6. INITIALIZE TRAINER
    # ====================================================
    trainer = ASRTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        processor=processor  # Penting untuk decoding saat validasi (WER/CER)
    )

    # ====================================================
    # 7. START TRAINING Loop
    # ====================================================
    print("\n" + "="*50)
    print(f"🚀 STARTING TRAINING: {config['experiment']['project_name']}")
    print("="*50)
    
    try:
        trainer.train() # Memulai loop epoch di src/trainer.py
    except KeyboardInterrupt:
        print("\n[Main] Training interrupted by user. Saving current state...")
    
    # ====================================================
    # 8. SAVE FINAL MODEL
    # ====================================================
    print("\n[Main] Training Finished. Saving artifacts...")
    
    final_save_path = os.path.join(output_dir, "final_model")
    os.makedirs(final_save_path, exist_ok=True)
    
    # Simpan bobot (weights) model murni ala PyTorch
    torch.save(model.state_dict(), os.path.join(final_save_path, "pytorch_model.bin"))
    
    # Simpan Processor (Ini masih aman karena kita buatkan CustomProcessor)
    processor.save_pretrained(final_save_path)
    
    print(f"✅ Model PyTorch & Processor diamankan di: {final_save_path}")
    print(f"✅ Gunakan path ini untuk script inference.py nanti.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR Training Script")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    main(args)