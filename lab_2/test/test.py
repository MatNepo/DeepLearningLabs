import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from sklearn.preprocessing import LabelEncoder

# Load data
data = pd.read_csv('D:\\Users\\Legion\\datasets\\emotions.csv')
texts = data['text'].values

# Encode labels as integers if they are not already
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(data['label'].values)

# Tokenization and vocabulary building
tokenizer = get_tokenizer("basic_english")
vocab = build_vocab_from_iterator(map(tokenizer, texts), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])  # For unknown tokens

# Convert text to numerical form
text_pipeline = lambda x: vocab(tokenizer(x))
tokenized_texts = [text_pipeline(text) for text in texts]

# Padding sequences
tokenized_texts = [torch.tensor(seq, dtype=torch.long) for seq in tokenized_texts]
padded_texts = pad_sequence(tokenized_texts, batch_first=True)
labels = torch.tensor(labels, dtype=torch.long)

# Create DataLoader
dataset = TensorDataset(padded_texts, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Print a sample to verify
for batch_texts, batch_labels in dataloader:
    print(batch_texts, batch_labels)
    break
