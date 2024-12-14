import torch
import torch.nn as nn

class MusicLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, fc_dim, device):
        super(MusicLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.ModuleList([
                nn.LSTM(embedding_dim if i == 0 else hidden_dim, hidden_dim, num_layers = 1, batch_first = True)
            for i in range(num_layers)
        ])
        self.fc1 = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.device = device
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, x, hidden = None, **kwargs):
        if hidden == None:
            hidden = self.init_hidden(x.shape[0])
        x = self.embedding(x)
        for i, lstm_layer in enumerate(self.lstm):
            x, hidden = lstm_layer(x, hidden)
            x = self.dropout(x)
        out = self.fc1(x)
        return out

    def init_hidden(self, batch_size):
            return (torch.zeros(1, batch_size, self.hidden_dim).to(self.device),
                    torch.zeros(1, batch_size, self.hidden_dim).to(self.device))
