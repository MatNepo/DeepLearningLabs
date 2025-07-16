import torch


class LMUCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, device):
        super(LMUCell, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.memory_size = memory_size

        self.A = torch.nn.Parameter(torch.randn(memory_size, memory_size).to(device))
        self.B = torch.nn.Parameter(torch.randn(memory_size, input_size).to(device))
        self.C = torch.nn.Parameter(torch.randn(hidden_size, memory_size).to(device))
        self.D = torch.nn.Parameter(torch.randn(hidden_size, input_size).to(device))

        self.input_layer = torch.nn.Linear(input_size, hidden_size).to(device)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()  # size: (batch_size, seq_length, input_size)
        c = torch.zeros(batch_size, self.memory_size, device=self.device)
        h = torch.zeros(batch_size, self.hidden_size, device=self.device)

        for t in range(seq_len):
            u = torch.tanh(self.input_layer(x[:, t]))  # input size -> hidden size
            c = self.A @ c + self.B @ u
            h = torch.tanh(self.C @ c + self.D @ u)  # Check dimensions here

        return h
