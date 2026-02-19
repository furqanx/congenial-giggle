import os
import json
import textgrid
from tqdm import tqdm
from pathlib import Path

# KONFIGURASI PATH (Sesuaikan dengan folder Anda)
DIRS = {
    "train": {
        "audio_dir": "data/interim/mfa_corpus_train",
        "textgrid_dir": "data/processed/mfa_aligned_train",
        "output_path": "data/processed/train_manifest.jsonl"
    },
    "val": {
        "audio_dir": "data/interim/mfa_corpus_val",
        "textgrid_dir": "data/processed/mfa_aligned_val",
        "output_path": "data/processed/val_manifest.jsonl"
    }
}

def parse_textgrid(tg_path):
    """Mengekstrak kata dan waktu dari file TextGrid"""
    try:
        tg = textgrid.TextGrid.fromFile(tg_path)
        words_tier = tg[0] # Biasanya tier 0 adalah 'words', tier 1 adalah 'phones'
        
        word_labels = []
        full_text = []
        
        for interval in words_tier:
            word = interval.mark
            # MFA menandai diam sebagai kosong "" atau "<sil>"
            if word and word != "<sil>" and word != "":
                start = interval.minTime
                end = interval.maxTime
                word_labels.append({
                    "word": word,
                    "start": round(start, 4),
                    "end": round(end, 4)
                })
                full_text.append(word)
                
        return word_labels, " ".join(full_text)
    except Exception as e:
        print(f"Error parsing {tg_path}: {e}")
        return None, None

def main():
    for split, paths in DIRS.items():
        print(f"🚀 Memproses set: {split.upper()}...")
        
        audio_dir = Path(paths["audio_dir"])
        tg_dir = Path(paths["textgrid_dir"])
        data_entries = []
        
        # Ambil semua file TextGrid yang berhasil dibuat MFA
        tg_files = list(tg_dir.glob("*.TextGrid"))
        
        for tg_file in tqdm(tg_files):
            file_id = tg_file.stem # Nama file tanpa ekstensi
            
            # Pastikan file audio pasangannya ada
            audio_path = audio_dir / f"{file_id}.flac"
            if not audio_path.exists():
                continue
                
            # Ekstrak Info
            words, transcript = parse_textgrid(str(tg_file))
            
            if words: # Hanya simpan jika ada kata terdeteksi
                entry = {
                    "id": file_id,
                    "audio_filepath": str(audio_path.absolute()), # Gunakan absolute path agar aman
                    "text": transcript,
                    "duration": words[-1]['end'], # Estimasi durasi dari kata terakhir
                    "labels": words # INI KUNCI JAWABAN KITA
                }
                data_entries.append(entry)

        # Simpan ke JSONL
        print(f"💾 Menyimpan {len(data_entries)} data ke {paths['output_path']}...")
        with open(paths['output_path'], 'w') as f:
            for entry in data_entries:
                json.dump(entry, f)
                f.write('\n')
                
    print("\n✅ SELESAI! Manifest siap digunakan untuk Training.")

if __name__ == "__main__":
    main()