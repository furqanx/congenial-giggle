import torch
import torch.nn as nn
import os
import numpy as np
from tqdm.auto import tqdm
from src.utils.metrics import compute_asr_metrics

class ASRTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, device, config, processor):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.processor = processor
        
        self.output_dir = os.path.join(config['experiment']['output_dir'], config['experiment']['project_name'])
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
        self.accum_steps = config['train'].get('gradient_accumulation_steps', 1)

        # ==========================================
        # SISTEM PEMBIDIK (LOSS FUNCTIONS)
        # ==========================================
        self.pad_id = self.processor.tokenizer.pad_token_id
        
        # 1. CTC Loss (Sang Penyelaras Waktu)
        # zero_infinity=True mencegah training crash jika ada audio yang durasinya
        # lebih pendek dari jumlah teks transcript-nya.
        self.ctc_loss_fn = nn.CTCLoss(blank=self.pad_id, zero_infinity=True)
        
        # 2. Attention Loss + Label Smoothing 0.1 (Sang Anti-Noise)
        # ignore_index=-100 memastikan token padding tidak ikut dihitung penaltinya
        self.attn_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=-100)
        
        # 3. Kalibrasi Hybrid (Biasanya 0.3 untuk CTC, 0.7 untuk Attention)
        self.alpha = config['train'].get('ctc_weight', 0.3)

    def train(self):
        epochs = self.config['train']['epochs']
        best_wer = float('inf')
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            train_loss = self._train_epoch(epoch)
            val_metrics = self.validate()
            
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val WER: {val_metrics['wer']:.4f} | Val CER: {val_metrics['cer']:.4f}")
            
            self.save_checkpoint(epoch, val_metrics, is_best=False)
            
            if val_metrics['wer'] < best_wer:
                print(f"🔥 Target Tertembak! New Best Model (WER: {best_wer:.4f} -> {val_metrics['wer']:.4f})")
                best_wer = val_metrics['wer']
                self.save_checkpoint(epoch, val_metrics, is_best=True)

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for step, batch in enumerate(pbar):
            input_values = batch['input_values'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # --- KALKULASI PANJANG AKTUAL (Tanpa Padding) ---
            # Menghitung durasi asli audio (asumsi padding adalah nilai 0.0)
            input_lengths = torch.sum(input_values[:, :, 0] != 0.0, dim=1).long()
            # Menghitung panjang teks asli (mengabaikan -100)
            target_lengths = torch.sum(labels != -100, dim=1).long()
            
            # Ekstrak label asli untuk CTC (buang semua angka -100 menjadi 1D array)
            ctc_targets = labels[labels != -100]

            with torch.cuda.amp.autocast(enabled=(self.scaler is not None)):
                # Tembak ke Model
                outputs = self.model(input_values, input_lengths, target_tokens=labels)
                ctc_logits = outputs["ctc_logits"]
                decoder_logits = outputs["decoder_logits"]

                # --- 1. Hitung CTC Loss ---
                # Syarat mutlak PyTorch CTC: Logits harus berbentuk (Waktu, Batch, Vocab) + LogSoftmax
                ctc_logits_t = ctc_logits.transpose(0, 1).log_softmax(2)
                loss_ctc = self.ctc_loss_fn(ctc_logits_t, ctc_targets, input_lengths, target_lengths)

                # --- 2. Hitung Attention Loss ---
                # Syarat mutlak CrossEntropy: Logits (Batch*Waktu, Vocab), Label (Batch*Waktu)
                loss_attn = self.attn_loss_fn(
                    decoder_logits.reshape(-1, decoder_logits.size(-1)), 
                    labels.reshape(-1)
                )

                # --- 3. Penggabungan Hibrida ---
                loss = (self.alpha * loss_ctc) + ((1 - self.alpha) * loss_attn)
                loss = loss / self.accum_steps

            # Backward Pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimizer Step
            if (step + 1) % self.accum_steps == 0:
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                if self.scheduler:
                    self.scheduler.step()
                
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.accum_steps
            pbar.set_postfix({'loss': loss.item() * self.accum_steps, 'lr': self.optimizer.param_groups[0]['lr']})

        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        print("Running Validation (CTC Decoding)...")
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", leave=False):
                input_values = batch['input_values'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                input_lengths = torch.sum(input_values[:, :, 0] != 0.0, dim=1).long()
                
                # Saat validasi, JANGAN berikan target_tokens agar model tidak nyontek
                outputs = self.model(input_values, input_lengths, target_tokens=None)
                
                # Untuk validasi cepat, kita gunakan prediksi dari CTC Head (Argmax)
                ctc_logits = outputs["ctc_logits"]
                pred_ids = torch.argmax(ctc_logits, dim=-1)
                
                # Decode prediksi
                pred_str = self.processor.batch_decode(pred_ids)
                
                # Decode Label (tangani -100)
                labels[labels == -100] = self.pad_id
                label_str = self.processor.batch_decode(labels, group_tokens=False)
                
                all_preds.extend(pred_str)
                all_labels.extend(label_str)

        metrics = compute_asr_metrics(all_preds, all_labels)
        return metrics

    def save_checkpoint(self, epoch, metrics, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config
        }
        
        filename = "checkpoint_best.pth" if is_best else "checkpoint_last.pth"
        save_path = os.path.join(self.output_dir, filename)
        torch.save(checkpoint, save_path)
        
        # Processor HuggingFace tetap bisa disave secara standar
        if is_best:
            self.processor.save_pretrained(os.path.join(self.output_dir, "best_processor"))

    def load_checkpoint(self, checkpoint_path):
        print(f"Memuat amunisi dari {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        start_epoch = checkpoint['epoch'] + 1
        return start_epoch