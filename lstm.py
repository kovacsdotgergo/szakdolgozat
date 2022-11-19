import torch
import torch.nn as nn

class LSTM_mel(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, output_size, have_cuda):
    super().__init__()
    self.num_layers = num_layers
    self.hidden_size = hidden_size
    self.lstm = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size,
                        num_layers=self.num_layers, batch_first=True)
    self.fc = nn.Linear(self.hidden_size, output_size)
    self.have_cuda = have_cuda

  def forward(self, x):
    """@param[in]   x   input size: (batch, time_windows, mels)
    """
    h_0 = torch.autograd.Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
    c_0 = torch.autograd.Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
    if self.have_cuda:
        h_0 = h_0.cuda()
        c_0 = c_0.cuda()
    #x = x.transpose(1, 2)
    out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
    h_n = h_n.view(-1, self.hidden_size)
    return self.fc(h_n)