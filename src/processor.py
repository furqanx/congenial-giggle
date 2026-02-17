import os
import json
import pandas as pd
from transformers import Wav2Vec2CTCTokenizer

def create_vocabulary_from_data(
        train_manifest_path, 
        val_manifest_path, 
        output_dir
    ):
    """
    Membaca semua teks di dataset train & val, lalu membuat vocab.json
    """
    print("[Tokenizer] Mengintai kosakata dari dataset...")
    
    df_train = pd.read_json(train_manifest_path, lines=True)
    df_val = pd.read_json(val_manifest_path, lines=True)
    
    all_text = pd.concat([df_train['orthographic_text'], df_val['orthographic_text']])

    vocab_set = set()
    for text in all_text:
        if isinstance(text, str):
            vocab_set.update(list(text))
    
    vocab_dict = {v: k for k, v in enumerate(sorted(list(vocab_set)))}
    
    if " " in vocab_dict:
        vocab_dict["|"] = vocab_dict.pop(" ")
    
    # --- PENTING: AMUNISI TOKEN SPESIAL ---
    vocab_dict["[UNK]"] = len(vocab_dict) # Unkown character
    vocab_dict["[PAD]"] = len(vocab_dict) # Padding (Untuk menyamakan panjang batch)
    vocab_dict["<s>"] = len(vocab_dict)   # Start of Sentence (Wajib untuk Decoder)
    vocab_dict["</s>"] = len(vocab_dict)  # End of Sentence (Wajib untuk Decoder)

    os.makedirs(output_dir, exist_ok=True)
    vocab_path = os.path.join(output_dir, "vocab.json")
    
    with open(vocab_path, 'w') as f:
        json.dump(vocab_dict, f)
        
    print(f"[Tokenizer] Kamus berhasil dibuat: {vocab_path} ({len(vocab_dict)} token)")
    return vocab_path

class CustomProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def save_pretrained(self, save_dir):
        self.tokenizer.save_pretrained(save_dir)

def get_processor(config):
    """
    Membuat atau Memuat Tokenizer.
    """
    save_dir = os.path.join(config['experiment']['output_dir'], "processor")
    
    if os.path.exists(save_dir) and os.path.exists(os.path.join(save_dir, "vocab.json")):
        print(f"[Tokenizer] Memuat kamus yang sudah ada dari {save_dir}")
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(save_dir)
        return CustomProcessor(tokenizer)

    # JIKA BELUM ADA, BUAT DARI NOL
    vocab_path = create_vocabulary_from_data(
        config['data']['train_manifest'],
        config['data']['val_manifest'],
        save_dir
    )
    # Output
    # {
    #   "a": 0,
    #   "b": 1,
    #   "c": 2,
    #   "d": 3,
    #   "...": "...",
    #   "|": 27,
    #   "[UNK]": 28,
    #   "[PAD]": 29,
    #   "<s>": 30,
    #   "</s>": 31
    # }
    
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path, 
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="<s>",   # Daftarkan token awal
        eos_token="</s>",  # Daftarkan token akhir
        word_delimiter_token="|" 
    )
    
    processor = CustomProcessor(tokenizer)
    processor.save_pretrained(save_dir)
    print("[Tokenizer] Pembuatan kamus selesai dan diamankan.")
    
    return processor