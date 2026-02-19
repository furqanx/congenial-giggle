import torch
import torch.nn as nn
import torchaudio
import math
from transformers import AutoModel

# ==========================================
# BLOK 0: KOMPONEN PENDUKUNG
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), 0, :]
        return self.dropout(x)

# ==========================================
# BLOK 1: THE BRAIN (Contextual Encoders)
# ==========================================
class BiLSTMEncoder(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, num_layers=3, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )
        self.output_dim = hidden_dim * 2

    def forward(self, x, lengths=None):
        out, _ = self.lstm(x)
        return out

class ConformerEncoder(nn.Module):
    def __init__(self, input_dim=768, num_heads=8, ffn_dim=1024, num_layers=4, dropout=0.1):
        super().__init__()
        self.conformer = torchaudio.models.Conformer(
            input_dim=input_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=31,
            dropout=dropout
        )
        self.output_dim = input_dim

    def forward(self, x, lengths):
        out, _ = self.conformer(x, lengths)
        return out

# ==========================================
# BLOK 2: THE TRANSLATOR (Decoder)
# ==========================================
class TransformerDecoderModule(nn.Module):
    def __init__(self, vocab_size, encoder_dim, hidden_dim=512, num_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 4, 
            dropout=dropout, 
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.encoder_proj = nn.Linear(encoder_dim, hidden_dim) if encoder_dim != hidden_dim else nn.Identity()
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tgt, memory, tgt_mask=None, tgt_pad_mask=None):
        tgt_emb = self.pos_encoder(self.embedding(tgt))
        memory_proj = self.encoder_proj(memory) 
        
        out = self.decoder(
            tgt=tgt_emb,
            memory=memory_proj,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask
        )
        return self.output_proj(out)

# ==========================================
# BLOK 3: THE END-TO-END HYBRID CANGKANG (WavLM + Encoder + Decoder)
# ==========================================
class EndToEndHybridASR(nn.Module):
    def __init__(self, vocab_size, pad_token_id, config):
        super().__init__()
        self.pad_token_id = pad_token_id
        
        # --- 1. THE EARS: WavLM/Wav2Vec2 Backbone ---
        backbone_name = config.get("backbone", "microsoft/wavlm-base-plus")
        print(f"[Model Factory] Mengunduh/Memuat Telinga Bionik: {backbone_name}")
        self.backbone = AutoModel.from_pretrained(backbone_name)
        
        if config.get("freeze_backbone", False):
            print("[Model Factory] Membekukan bobot Backbone (Feature Extraction Mode)")
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Ambil dimensi output bawaan dari backbone (biasanya 768 untuk base model)
        backbone_dim = self.backbone.config.hidden_size
        
        # --- 2. THE BRAIN: Contextual Encoder ---
        encoder_type = config.get("encoder_type", "bilstm").lower()
        print(f"[Model Factory] Merakit Otak: {encoder_type.upper()}")
        
        if encoder_type == "bilstm":
            self.encoder = BiLSTMEncoder(
                input_dim=backbone_dim, 
                hidden_dim=config.get("hidden_dim", 512), 
                num_layers=config.get("num_enc_layers", 3),
                dropout=config.get("dropout", 0.1)
            )
        elif encoder_type == "conformer":
            self.encoder = ConformerEncoder(
                input_dim=backbone_dim,
                num_heads=config.get("num_heads", 8),
                num_layers=config.get("num_enc_layers", 4),
                dropout=config.get("dropout", 0.1)
            )
        else:
            raise ValueError("encoder_type harus 'bilstm' atau 'conformer'")
            
        # --- 3. CTC HEAD ---
        self.ctc_head = nn.Linear(self.encoder.output_dim, vocab_size)
        
        # --- 4. ATTENTION DECODER ---
        self.decoder = TransformerDecoderModule(
            vocab_size=vocab_size, 
            encoder_dim=self.encoder.output_dim,
            hidden_dim=config.get("decoder_hidden_dim", 512),
            num_heads=config.get("num_heads", 8),
            num_layers=config.get("num_dec_layers", 2),
            dropout=config.get("dropout", 0.1)
        )

    def forward(self, input_values, target_tokens=None):
        """
        input_values: Audio gelombang mentah (Batch Size, Panjang Audio)
        target_tokens: Teks transcript (Batch Size, Max_Text_Len)
        """
        # --- 1. JALUR BACKBONE (Audio -> Vektor) ---
        # WavLM otomatis membuang padding dan mengekstrak fitur
        outputs = self.backbone(input_values)
        acoustic_features = outputs.last_hidden_state # Shape: (Batch Size, Waktu, 768)
        
        # Karena kita menggunakan WavLM via HuggingFace tanpa padding mask khusus, 
        # kita estimasikan panjang sequence penuh untuk input ke encoder.
        batch_size, seq_len, _ = acoustic_features.size()
        input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=input_values.device)
        
        # --- 2. JALUR ENCODER ---
        if isinstance(self.encoder, ConformerEncoder):
            memory = self.encoder(acoustic_features, input_lengths)
        else:
            memory = self.encoder(acoustic_features)
            
        # --- 3. JALUR CTC ---
        ctc_logits = self.ctc_head(memory) 
        
        # --- 4. JALUR ATTENTION DECODER ---
        decoder_logits = None
        if target_tokens is not None:
            tgt_seq_len = target_tokens.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(input_values.device)
            
            tgt_pad_mask = (target_tokens == self.pad_token_id) | (target_tokens == -100)
            safe_target_tokens = target_tokens.masked_fill(target_tokens == -100, self.pad_token_id)
            
            decoder_logits = self.decoder(
                tgt=safe_target_tokens, 
                memory=memory, 
                tgt_mask=tgt_mask,
                tgt_pad_mask=tgt_pad_mask
            )
            
        return {"ctc_logits": ctc_logits, "decoder_logits": decoder_logits, "input_lengths": input_lengths}

# ==========================================
# FACTORY FUNCTION
# ==========================================
def build_model(config, processor):
    vocab_size = len(processor.tokenizer)
    pad_token_id = processor.tokenizer.pad_token_id
    
    model = EndToEndHybridASR(vocab_size=vocab_size, pad_token_id=pad_token_id, config=config)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Total Senjata (Trainable Parameters): {trainable_params:,}")
    
    return model