# Simpler and smaller setup for Seq2Seq with Attention, Training, Evaluation, and Plotting

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import accuracy_score

# Set seed and device
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dummy dataset
SRC_VOCAB_SIZE = 15
TRG_VOCAB_SIZE = 15
SEQ_LENGTH = 6
NUM_SAMPLES = 100
BATCH_SIZE = 16

class DummyDataset(Dataset):
    def __init__(self, n_samples):
        self.samples = [(torch.randint(1, SRC_VOCAB_SIZE, (SEQ_LENGTH,)), 
                         torch.randint(1, TRG_VOCAB_SIZE, (SEQ_LENGTH,))) for _ in range(n_samples)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

train_dataset = DummyDataset(NUM_SAMPLES)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, hidden = self.rnn(embedded)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        return outputs, hidden.unsqueeze(0)

# Attention
class Attention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hidden_dim * 2 + dec_hidden_dim, dec_hidden_dim)
        self.v = nn.Linear(dec_hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        hidden = hidden.permute(1, 0, 2).repeat(1, encoder_outputs.size(1), 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        return F.softmax(self.v(energy).squeeze(2), dim=1)

# Decoder
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hidden_dim, dec_hidden_dim, attention):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim + enc_hidden_dim * 2, dec_hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(emb_dim + dec_hidden_dim + enc_hidden_dim * 2, output_dim)
        self.attention = attention

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(1)
        embedded = self.embedding(input)
        attn_weights = self.attention(hidden, encoder_outputs).unsqueeze(1)
        context = torch.bmm(attn_weights, encoder_outputs)
        rnn_input = torch.cat((embedded, context), dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        output = output.squeeze(1)
        context = context.squeeze(1)
        embedded = embedded.squeeze(1)
        return self.fc_out(torch.cat((output, context, embedded), dim=1)), hidden

# Seq2Seq model
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size, trg_len = trg.shape
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(device)
        encoder_outputs, hidden = self.encoder(src)
        input = trg[:, 0]
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[:, t] = output
            input = trg[:, t] if random.random() < teacher_forcing_ratio else output.argmax(1)
        return outputs

# Instantiate model
HID_DIM = 32
EMB_DIM = 16
attention = Attention(HID_DIM, HID_DIM)
encoder = Encoder(SRC_VOCAB_SIZE, EMB_DIM, HID_DIM)
decoder = Decoder(TRG_VOCAB_SIZE, EMB_DIM, HID_DIM, HID_DIM, attention)
model = Seq2Seq(encoder, decoder).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
EPOCHS = 10
train_losses = []
accuracies = []

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    all_preds = []
    all_trues = []

    for src, trg in train_loader:
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        preds = output.argmax(1).detach().cpu().numpy()
        trues = trg.detach().cpu().numpy()
        all_preds.extend(preds)
        all_trues.extend(trues)

    acc = accuracy_score(all_trues, all_preds)
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    accuracies.append(acc)

# Plotting Loss and Accuracy
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, marker='o', label='Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(accuracies, marker='o', color='green', label='Accuracy')
plt.title('Token Accuracy')
plt.xlabel('Epoch')
plt.grid(True)
plt.tight_layout()
plt.show()

