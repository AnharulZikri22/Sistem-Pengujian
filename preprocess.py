import torch
import re
from transformers import BertTokenizer


class TextPreprocessor:
    def __init__(self, model_name="bert-base-uncased", max_len=50, device=None):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_len = max_len
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def clean_text(self, text):
        """Membersihkan teks dari karakter yang tidak diinginkan."""
        text = re.sub(
            r"[A-Z]", lambda x: x.group(0).lower(), text
        )  # Konversi ke huruf kecil
        text = re.sub(r"[^a-z\s]", "", text)  # Hapus karakter selain huruf
        text = re.sub(r"\s+", " ", text).strip()  # Hapus spasi berlebih
        return text

    def preprocess_text(self, texts):
        """Melakukan tokenisasi dan padding untuk input ke BERT."""
        if isinstance(texts, str):
            texts = [texts]  # Jika input string, ubah menjadi list

        texts = [self.clean_text(text) for text in texts]  # Bersihkan teks

        encoding = self.tokenizer(
            texts,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return encoding["input_ids"].to(self.device), encoding["attention_mask"].to(
            self.device
        )
