import torch
import torch.nn as nn
import torchaudio
import math

# ==========================================
# BLOK 0: KOMPONEN PENDUKUNG
# ==========================================
class PositionalEncoding(nn.Module):
    """
    Suntikan pemahaman urutan waktu (Time-Awareness) untuk Transformer Decoder.
    Karena Transformer membaca semua sekaligus, ia butuh "stempel waktu" di setiap kata.
    """
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
        # x shape: (Batch, Waktu, Dimensi)
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)

# ==========================================
# BLOK 1: ENCODERS (Sang Pengekstrak Fitur Akustik)
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
        self.output_dim = hidden_dim * 2 # Karena Bidirectional (Maju + Mundur)

    def forward(self, x, lengths=None):
        out, _ = self.lstm(x)
        return out

class ConformerEncoder(nn.Module):
    def __init__(self, input_dim=768, num_heads=8, ffn_dim=1024, num_layers=4, dropout=0.1):
        super().__init__()
        # Menggunakan Conformer bawaan Torchaudio yang sangat teroptimasi
        self.conformer = torchaudio.models.Conformer(
            input_dim=input_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=31,
            dropout=dropout
        )
        self.output_dim = input_dim # Conformer mempertahankan dimensi input

    def forward(self, x, lengths):
        # Conformer Torchaudio butuh informasi panjang masing-masing audio dalam batch
        out, _ = self.conformer(x, lengths)
        return out

# ==========================================
# BLOK 2: DECODER (Sang Model Bahasa / Ahli Tata Bahasa)
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
        
        # Jembatan penyesuai dimensi antara Encoder (Misal: 1024) ke Decoder (512)
        self.encoder_proj = nn.Linear(encoder_dim, hidden_dim) if encoder_dim != hidden_dim else nn.Identity()
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tgt, memory, tgt_mask=None, tgt_pad_mask=None):
        # tgt: Teks (Batch, Panjang Teks)
        # memory: Output dari Encoder (Batch, Waktu, Dimensi)
        
        tgt_emb = self.embedding(tgt) 
        tgt_emb = self.pos_encoder(tgt_emb)
        
        memory_proj = self.encoder_proj(memory) 
        
        out = self.decoder(
            tgt=tgt_emb,
            memory=memory_proj,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask
        )
        
        return self.output_proj(out)

# ==========================================
# BLOK 3: CANGKANG UTAMA (The Hybrid Wrapper)
# ==========================================
class HybridASRModel(nn.Module):
    def __init__(self, vocab_size, pad_token_id, config):
        super().__init__()
        self.pad_token_id = pad_token_id
        
        encoder_type = config.get("encoder_type", "bilstm").lower()
        input_dim = config.get("input_dim", 768) # Dimensi WavLM/HuBERT
        
        print(f"[Model Factory] Merakit senjata: {encoder_type.upper()} + Transformer Decoder")
        
        # 1. SETUP ENCODER
        if encoder_type == "bilstm":
            self.encoder = BiLSTMEncoder(
                input_dim=input_dim, 
                hidden_dim=config.get("hidden_dim", 512), 
                num_layers=config.get("num_enc_layers", 3),
                dropout=config.get("dropout", 0.1)
            )
        elif encoder_type == "conformer":
            self.encoder = ConformerEncoder(
                input_dim=input_dim,
                num_heads=config.get("num_heads", 8),
                num_layers=config.get("num_enc_layers", 4),
                dropout=config.get("dropout", 0.1)
            )
        else:
            raise ValueError("encoder_type di config.yaml harus 'bilstm' atau 'conformer'")
            
        # 2. CTC HEAD (Jalur Tembakan Pertama)
        self.ctc_head = nn.Linear(self.encoder.output_dim, vocab_size)
        
        # 3. ATTENTION DECODER (Jalur Tembakan Kedua)
        self.decoder = TransformerDecoderModule(
            vocab_size=vocab_size, 
            encoder_dim=self.encoder.output_dim,
            hidden_dim=config.get("decoder_hidden_dim", 512),
            num_heads=config.get("num_heads", 8),
            num_layers=config.get("num_dec_layers", 2),
            dropout=config.get("dropout", 0.1)
        )

    def forward(self, input_values, input_lengths, target_tokens=None):
        """
        input_values: (Batch, Max_Time, 768) - Dari HDF5 Dataloader
        input_lengths: (Batch) - Panjang waktu asli sebelum di-padding
        target_tokens: (Batch, Max_Text_Len) - Teks transcript untuk melatih Decoder
        """
        
        # --- 1. JALUR ENCODER ---
        if isinstance(self.encoder, ConformerEncoder):
            memory = self.encoder(input_values, input_lengths)
        else:
            memory = self.encoder(input_values)
            
        # --- 2. JALUR CTC ---
        # Logits ini akan dimakan oleh nn.CTCLoss nanti di trainer.py
        ctc_logits = self.ctc_head(memory) 
        
        # --- 3. JALUR ATTENTION (Hanya jalan saat Training / target_tokens ada) ---
        decoder_logits = None
        if target_tokens is not None:
            # Cegah model "mencontek" kata di masa depan menggunakan Causal Mask
            tgt_seq_len = target_tokens.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(input_values.device)
            
            # Buat Pad Mask agar padding teks (token -100) diabaikan oleh Attention
            tgt_pad_mask = (target_tokens == self.pad_token_id)
            # Karena Dataloader kita mengubah pad menjadi -100, kita tangani juga
            tgt_pad_mask = tgt_pad_mask | (target_tokens == -100)
            
            # Ubah -100 kembali ke pad_token_id agar tidak crash saat masuk layer Embedding
            safe_target_tokens = target_tokens.masked_fill(target_tokens == -100, self.pad_token_id)
            
            decoder_logits = self.decoder(
                tgt=safe_target_tokens, 
                memory=memory, 
                tgt_mask=tgt_mask,
                tgt_pad_mask=tgt_pad_mask
            )
            
        return {"ctc_logits": ctc_logits, "decoder_logits": decoder_logits}

# ==========================================
# FACTORY FUNCTION (Untuk dipanggil di main.py / trainer.py)
# ==========================================
def build_model(config, processor):
    vocab_size = len(processor.tokenizer)
    pad_token_id = processor.tokenizer.pad_token_id
    
    model = HybridASRModel(vocab_size=vocab_size, pad_token_id=pad_token_id, config=config)
    
    # Cetak parameter untuk laporan intelijen
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Total Senjata (Trainable Parameters): {trainable_params:,}")
    
    return model