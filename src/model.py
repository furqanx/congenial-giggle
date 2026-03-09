import torch
import torch.nn as nn
from transformers import AutoModelForCTC, AutoConfig

class EndToEndCTCASR(nn.Module):
    def __init__(self, vocab_size, pad_token_id, config):
        super().__init__()
        self.pad_token_id = pad_token_id
        
        backbone_name = config.get("backbone", "microsoft/wavlm-base-plus")
        print(f"[Model Factory] Merakit Model Murni CTC dengan Backbone: {backbone_name}")
        
        hf_config = AutoConfig.from_pretrained(
            backbone_name,
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            ctc_loss_reduction="mean",
            mask_time_prob=config.get("mask_time_prob", 0.05), 
            mask_time_length=config.get("mask_time_length", 10),
            mask_feature_prob=config.get("mask_feature_prob", 0.05),
            mask_feature_length=config.get("mask_feature_length", 10)
        )
        
        self.backbone = AutoModelForCTC.from_pretrained(
            backbone_name, 
            config=hf_config,
            ignore_mismatched_sizes=True 
        )
        
        if config.get("freeze_feature_extractor", True):
            print("[Model Factory] Membekukan CNN Feature Encoder (Sangat Disarankan)")
            if hasattr(self.backbone, "freeze_feature_encoder"):
                self.backbone.freeze_feature_encoder()
            else:
                self.backbone.freeze_feature_extractor()

    # REVISI KRITIS: Terima argumen 'labels' dan lemparkan ke backbone Hugging Face
    def forward(self, input_values, attention_mask=None, labels=None, **kwargs):
        
        # Biarkan Hugging Face yang melakukan semua perhitungan berat (termasuk CTC Loss)
        outputs = self.backbone(
            input_values=input_values,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Kembalikan dictionary sesuai dengan ekspektasi trainer.py kita yang baru
        return {
            "loss": outputs.loss if labels is not None else None,
            "ctc_logits": outputs.logits 
        }

# ==========================================
# FACTORY FUNCTION
# ==========================================
def build_model(config, processor):
    vocab_size = len(processor.tokenizer)
    pad_token_id = processor.tokenizer.pad_token_id
    
    model = EndToEndCTCASR(vocab_size=vocab_size, pad_token_id=pad_token_id, config=config)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Total Senjata (Trainable Parameters): {trainable_params:,}")
    
    return model