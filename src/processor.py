import os
import re
import json
import pandas as pd
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
import inflect

p = inflect.engine()

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    
    # 1. HAPUS ANGKA ANOTASI (misal: "running2" -> "running", "wrong2" -> "wrong")
    # Regex ini mencari huruf yang langsung diikuti angka, dan membuang angkanya.
    text = re.sub(r'([a-z]+)\d+', r'\1', text)
    
    # 2. SULAP ANGKA STANDALONE MENJADI KATA (misal: "90" -> "ninety", "2" -> "two")
    def convert_num(match):
        num_str = match.group()
        word = p.number_to_words(num_str)
        # Bersihkan strip dan koma bawaan inflect (misal: twenty-five -> twenty five)
        word = word.replace('-', ' ').replace(',', '')
        return " " + word + " "
        
    # Regex \b\d+\b hanya akan menangkap angka yang terpisah oleh spasi
    text = re.sub(r'\b\d+\b', convert_num, text)
        
    # 3. Petakan karakter fonetik/aksen ke alfabet terdekat
    char_map = {
        'é': 'e', 'æ': 'ae', '\x92': "'", 
        'ɑ': 'a', 'ɔ': 'o', 'ɜ': 'e', 'ɪ': 'i', 'ʊ': 'u', 'ʌ': 'u'
    }
    for k, v in char_map.items():
        text = text.replace(k, v)
        
    # 4. Hapus SEMUA tanda baca (hanya sisakan huruf a-z dan spasi)
    text = re.sub(r'[^a-z\s]', '', text)
    
    # 5. Rapikan spasi ganda menjadi spasi tunggal
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

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
    
    # PERBAIKAN: Menggunakan key 'text' sesuai output MFA kita
    all_text = pd.concat([df_train['orthographic_text'], df_val['orthographic_text']])

    vocab_set = set()
    for text in all_text:
        clean_text = normalize_text(text)  ###
        vocab_set.update(list(text))
    
    vocab_dict = {v: k for k, v in enumerate(sorted(list(vocab_set)))}
    
    if " " in vocab_dict:
        vocab_dict["|"] = vocab_dict.pop(" ")
    
    # --- PENTING: AMUNISI TOKEN SPESIAL ---
    vocab_dict["[UNK]"] = len(vocab_dict) # Unknown character
    vocab_dict["[PAD]"] = len(vocab_dict) # Padding (Juga berfungsi sebagai CTC Blank Token)
    # vocab_dict["<s>"] = len(vocab_dict)   # Start of Sentence (Wajib untuk Decoder) ##
    # vocab_dict["</s>"] = len(vocab_dict)  # End of Sentence (Wajib untuk Decoder)   ##

    os.makedirs(output_dir, exist_ok=True)
    vocab_path = os.path.join(output_dir, "vocab.json")
    
    with open(vocab_path, 'w') as f:
        json.dump(vocab_dict, f)
        
    print(f"[Tokenizer] Kamus berhasil dibuat: {vocab_path} ({len(vocab_dict)} token)")
    return vocab_path


def get_processor(config):
    """
    Membuat atau Memuat Processor (FeatureExtractor + Tokenizer).
    """
    save_dir = os.path.join(config['experiment']['output_dir'], "processor")
    
    # Tambahkan Feature Extractor untuk membersihkan & menormalisasi Audio
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, 
        sampling_rate=config['data']['sample_rate'], 
        padding_value=0.0, 
        do_normalize=True, 
        return_attention_mask=True
    )

    if os.path.exists(save_dir) and os.path.exists(os.path.join(save_dir, "vocab.json")):
        print(f"[Processor] Memuat processor utuh yang sudah ada dari {save_dir}")
        processor = Wav2Vec2Processor.from_pretrained(save_dir)
        return processor

    # JIKA BELUM ADA, BUAT DARI NOL
    vocab_path = create_vocabulary_from_data(
        config['data']['train_manifest'],
        config['data']['val_manifest'],
        save_dir
    )
    
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path, 
        unk_token="[UNK]",
        pad_token="[PAD]",
        # bos_token="<s>",   ####
        # eos_token="</s>",  ####
        word_delimiter_token="|" 
    )
    
    # GABUNGKAN Feature Extractor dan Tokenizer menjadi satu Processor resmi
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    processor.save_pretrained(save_dir)
    print("[Processor] Pembuatan processor selesai dan diamankan.")
    
    return processor