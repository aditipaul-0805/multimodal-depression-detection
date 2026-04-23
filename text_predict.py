import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load model & tokenizer
model = DistilBertForSequenceClassification.from_pretrained("../../models/text_model")
tokenizer = DistilBertTokenizer.from_pretrained("../../models/text_model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def predict_text(text):
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()

    if prediction == 1:
        return "Positive / Non-depressed"
    else:
        return "Negative / Depressed"
    
def predict_text(text):
    ...
    return "Negative / Depressed"

# TEST BLOCK
if __name__ == "__main__":
    text = "I feel very lonely and tired"
    result = predict_text(text)
    print("📝 Text Emotion:", result)    
