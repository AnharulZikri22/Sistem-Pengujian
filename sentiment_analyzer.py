import torch
import torch.nn.functional as F
from transformers import BertForSequenceClassification
from preprocess import TextPreprocessor


class SentimentAnalyzer:
    def __init__(self, model_path, model_name="bert-base-uncased", device=None):
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.preprocessor = TextPreprocessor(model_name, device=self.device)

        # Load model
        self.model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text):
        """Memprediksi sentimen dari teks input."""
        input_ids, attention_mask = self.preprocessor.preprocess_text(text)
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask=attention_mask).logits
            probabilities = F.softmax(logits, dim=-1).cpu().numpy()[0]

        prob_negative, prob_positive = probabilities
        sentiment = "😊 Positive" if prob_positive > prob_negative else "😡 Negative"
        return sentiment, prob_negative, prob_positive
