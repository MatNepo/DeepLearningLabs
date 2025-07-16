# app.py
from fastapi import FastAPI, HTTPException
from transformers import BertTokenizer
import torch
from model import EmotionClassifier

app = FastAPI()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = EmotionClassifier(input_dim=128, hidden_dim=256, num_classes=5)
model.load_state_dict(torch.load('path/to/saved_model.pth'))
model.to(device)
model.eval()


@app.post("/predict")
async def predict(text: str):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        output = model(input_ids, attention_mask)
        prediction = torch.argmax(output, dim=1).item()

    return {"emotion": prediction}
