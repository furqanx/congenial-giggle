import torch
import torch.nn as nn
import os
from tqdm.auto import tqdm
from src.utils.metrics import compute_asr_metrics # Pastikan file ini ada

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
        
        # Mixed Precision untuk mempercepat training dan menghemat VRAM
        self.scaler = torch.amp.GradScaler() if torch.cuda.is_available() else None
        self.accum_steps = config['train'].get('gradient_accumulation_steps', 1)

        # ==========================================
        # SISTEM PEMBIDIK (LOSS FUNCTIONS)
        # ==========================================
        self.pad_id = self.processor.tokenizer.pad_token_id
        
        # 1. CTC Loss
        # zero_infinity=True mencegah inf/nan jika ada label yang lebih panjang dari output audio
        self.ctc_loss_fn = nn.CTCLoss(blank=self.pad_id, zero_infinity=True)
        
        # 2. Attention Loss
        # ignore_index=-100 agar token padding tidak dipenalti
        self.attn_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=-100)
        
        self.alpha = config['train'].get('ctc_weight', 0.3)

    def train(self):
        epochs = self.config['train']['epochs']
        best_wer = float('inf')
        
        print("\n[Trainer] Memulai Siklus Pelatihan...")
        for epoch in range(epochs):
            train_loss = self._train_epoch(epoch)
            
            # --- Validasi dilakukan setiap selesai 1 epoch ---
            print(f"\nEpoch {epoch+1} Selesai. Mengevaluasi Model...")
            val_metrics = self.validate()
            
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val WER: {val_metrics['wer']:.4f} | Val CER: {val_metrics['cer']:.4f}")
            
            self.save_checkpoint(epoch, val_metrics, is_best=False)
            
            if val_metrics['wer'] < best_wer:
                print(f"🔥 Rekor Baru! Menyimpan Model Terbaik (WER: {best_wer:.4f} -> {val_metrics['wer']:.4f})")
                best_wer = val_metrics['wer']
                self.save_checkpoint(epoch, val_metrics, is_best=True)

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} Training", leave=False)
        
        for step, batch in enumerate(pbar):
            input_values = batch['input_values'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Autocast untuk Mixed Precision (fp16)
            with torch.amp.autocast(device_type="cuda" if "cuda" in str(self.device) else "cpu", enabled=(self.scaler is not None)):
                # --- TEMBAK KE MODEL ---
                # Model akan mengembalikan logits dan panjang asli fiturnya
                outputs = self.model(input_values, target_tokens=labels)
                ctc_logits = outputs["ctc_logits"]       # Shape: (Batch, Waktu, Vocab)
                decoder_logits = outputs["decoder_logits"] # Shape: (Batch, Teks, Vocab)
                input_lengths = outputs["input_lengths"] # Shape: (Batch)

                # --- 1. PERSIAPAN DATA UNTUK CTC LOSS ---
                # CTC meminta logits ditranspose menjadi (Waktu, Batch, Vocab)
                ctc_logits_t = ctc_logits.transpose(0, 1).log_softmax(2)
                
                # Hitung panjang masing-masing teks tanpa menghitung padding (-100)
                target_lengths = torch.sum(labels != -100, dim=1).long()
                
                # PyTorch CTC meminta label dalam bentuk 1D Array (tanpa padding sama sekali)
                ctc_targets = labels[labels != -100]

                # Hitung CTC Loss
                loss_ctc = self.ctc_loss_fn(ctc_logits_t, ctc_targets, input_lengths, target_lengths)

                # --- 2. PERSIAPAN DATA UNTUK ATTENTION LOSS ---
                # Ratakan dimensi batch dan teks (Batch * Max_Teks, Vocab)
                vocab_size = decoder_logits.size(-1)
                flat_decoder_logits = decoder_logits.reshape(-1, vocab_size)
                flat_labels = labels.reshape(-1)
                
                loss_attn = self.attn_loss_fn(flat_decoder_logits, flat_labels)

                # --- 3. GABUNGKAN HYBRID LOSS ---
                loss = (self.alpha * loss_ctc) + ((1 - self.alpha) * loss_attn)
                loss = loss / self.accum_steps

            # Backward Pass menggunakan Scaler
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimizer Step (dengan Gradient Accumulation)
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
            pbar.set_postfix({'loss': loss.item() * self.accum_steps})

        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.val_loader, desc="Validating", leave=False)
        with torch.no_grad():
            for batch in pbar:
                input_values = batch['input_values'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Saat validasi, target_tokens harus None agar decoder memprediksi murni
                outputs = self.model(input_values, target_tokens=None)
                ctc_logits = outputs["ctc_logits"]
                
                # --- DECODING CTC KASAR (Greedy) ---
                pred_ids = torch.argmax(ctc_logits, dim=-1)
                
                # Kembalikan -100 ke ID padding agar tokenizer tidak error
                labels[labels == -100] = self.pad_id
                
                # Ubah ID Angka kembali menjadi String Teks
                pred_str = self.processor.batch_decode(pred_ids)
                label_str = self.processor.batch_decode(labels, group_tokens=False)
                
                all_preds.extend(pred_str)
                all_labels.extend(label_str)

        # Hitung skor WER/CER dari utils Anda
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
        
        if is_best:
            self.processor.save_pretrained(os.path.join(self.output_dir, "best_processor"))