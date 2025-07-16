import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch
from torch.nn.utils.rnn import pad_sequence

def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df[['text', 'label']]  # Keep only relevant columns
    df.columns = ['text', 'emotion']  # Rename columns for clarity
    return df

def tokenize_text(text):
    tokenizer = get_tokenizer("basic_english")
    return tokenizer(text)


def build_vocab(df):
    # Build vocabulary from training data
    vocab = build_vocab_from_iterator(map(tokenize_text, df['text']), specials=["<unk>", "<pad>"])
    vocab.set_default_index(vocab["<unk>"])  # Set the default index for unknown tokens
    return vocab

def pad_sequences(tokenized_texts, padding_value):
    # Pad the sequences to the same length
    return pad_sequence([torch.tensor(tokens) for tokens in tokenized_texts],
                        batch_first=True, padding_value=padding_value)

def preprocess_data(df):
    # Encode the labels
    label_encoder = LabelEncoder()
    df['emotion'] = label_encoder.fit_transform(df['emotion'])

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'].values, df['emotion'].values, test_size=0.2, random_state=42
    )

    # Build vocabulary from the training set
    vocab = build_vocab(pd.DataFrame(X_train, columns=['text']))

    # Tokenize and numericalize text
    X_train_numerical = [[vocab[token] for token in tokenize_text(text)] for text in X_train]
    X_test_numerical = [[vocab[token] for token in tokenize_text(text)] for text in X_test]

    # Pad sequences
    padding_value = vocab["<pad>"]
    X_train_padded = pad_sequences(X_train_numerical, padding_value)
    X_test_padded = pad_sequences(X_test_numerical, padding_value)

    return X_train_padded, X_test_padded, torch.tensor(y_train), torch.tensor(y_test), label_encoder, vocab
