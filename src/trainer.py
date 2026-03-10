import torch
import os
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
        
        # --- TAMBAHAN MULTI-GPU 1: Bungkus Model ---
        if torch.cuda.device_count() > 1:
            print(f"🔥 Mengaktifkan {torch.cuda.device_count()} GPU dengan DataParallel!")
            self.model = torch.nn.DataParallel(self.model)
        # ------------------------------------------
            
        self.output_dir = os.path.join(config['experiment']['output_dir'], config['experiment']['project_name'])
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Mixed Precision untuk mempercepat training dan menghemat VRAM
        self.scaler = torch.amp.GradScaler() if torch.cuda.is_available() else None
        self.accum_steps = config['train'].get('gradient_accumulation_steps', 1)

        self.pad_id = self.processor.tokenizer.pad_token_id

    def train(self):
        epochs = self.config['train']['epochs']
        best_wer = float('inf')
        
        print("\n[Trainer] Memulai Siklus Pelatihan Murni CTC...")
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
                outputs = self.model(input_values=input_values, labels=labels)
                
                loss = outputs["loss"]
                
                # --- TAMBAHAN MULTI-GPU 2: Rata-ratakan Loss ---
                if torch.cuda.device_count() > 1:
                    loss = loss.mean()
                # -----------------------------------------------
                
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
                
                # Saat validasi, jangan beri label agar tidak menghitung loss
                outputs = self.model(input_values=input_values)
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

        metrics = compute_asr_metrics(all_preds, all_labels)
        return metrics

    def save_checkpoint(self, epoch, metrics, is_best=False):
        # --- TAMBAHAN MULTI-GPU 3: Ambil state_dict asli tanpa prefix 'module.' ---
        if isinstance(self.model, torch.nn.DataParallel):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        # -------------------------------------------------------------------------
            
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
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