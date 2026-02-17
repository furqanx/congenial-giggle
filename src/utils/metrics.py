import numpy as np
import re

try:
    import jiwer
except ImportError:
    print("[Metrics] Warning: Library 'jiwer' belum terinstall. Metrik WER/CER akan gagal.")
    jiwer = None

def compute_asr_metrics(pred_strs, label_strs):
    """
    Menghitung standar metrik ASR: WER dan CER dengan dekontaminasi teks tingkat militer.
    """
    if jiwer is None:
        return {"wer": 1.0, "cer": 1.0, "wrr": 0.0}

    clean_preds = []
    clean_labels = []

    for p, l in zip(pred_strs, label_strs):
        # 1. Ganti delimiter '|' kembali menjadi spasi biasa
        p = p.replace('|', ' ')
        l = l.replace('|', ' ')

        # 2. Hancurkan token spesial yang "bocor" dari Tokenizer
        tokens_to_remove = ['[PAD]', '[UNK]', '<s>', '</s>']
        for t in tokens_to_remove:
            p = p.replace(t, '')
            l = l.replace(t, '')

        # 3. Normalisasi brutal: Hapus spasi ganda, jadikan huruf kecil, hapus spasi ujung
        p = re.sub(r'\s+', ' ', p).lower().strip()
        l = re.sub(r'\s+', ' ', l).lower().strip()

        clean_preds.append(p)
        clean_labels.append(l)

    # 4. Filter Penyelamat: jiwer akan crash jika Ground Truth (label) benar-benar kosong ""
    valid_preds = []
    valid_labels = []
    for p, l in zip(clean_preds, clean_labels):
        if len(l) > 0: # Pastikan ada teks di ground truth
            valid_preds.append(p)
            valid_labels.append(l)
            
    # Jika kebetulan semua label di batch ini kosong (sangat jarang terjadi)
    if len(valid_labels) == 0:
         return {"wer": 0.0, "cer": 0.0, "wrr": 1.0}

    # 5. Eksekusi Perhitungan Metrik
    wer_score = jiwer.wer(valid_labels, valid_preds)
    cer_score = jiwer.cer(valid_labels, valid_preds)
    wrr_score = 1.0 - wer_score

    return {
        "wer": wer_score,
        "cer": cer_score,
        "wrr": wrr_score
    }