import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, nonlinearity = 'tanh'):
        super().__init__()

        self.rnn = nn.RNN(in_size, hidden_size, batch_first = True, nonlinearity = nonlinearity)
        self.decoder = nn.Linear(hidden_size, out_size) # Linear decoder layer that maps the hidden states to outputs

    def forward(self, x):
        rnn_out,_ = self.rnn(x) # Running the RNN is as easy as passing te input to the object (BatchSize x Sequence Length x Input Dimension)


        batch_size, T, _ = rnn_out.shape
        rnn_out_reshaped = rnn_out.reshape(batch_size * T, -1) # We want to combine the batch and sequence dimension, since the decoder doesn't depend on time
        decoder_out = self.decoder(rnn_out_reshaped) # Pass the hidden states to the decoder
        return decoder_out.reshape(batch_size, T, -1), rnn_out # Output hidden states in case we want to analyze them
                # Seperate batch and sequence dimensions again
    