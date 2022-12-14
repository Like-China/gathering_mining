import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
import settings as constants


class Encoder(nn.Module):
    """
    The characteristic dimension of the input vector, input_size = args.vocab_size = 18864
    The dimension of the hidden layer vector, hidden_size = args.embedding_size= 256
    num_layers = args.hidden_size = 3
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirectional, embedding):

        super(Encoder, self).__init__()
        self.num_directions = 2 if bidirectional else 1
        assert hidden_size % self.num_directions == 0
        self.hidden_size = hidden_size // self.num_directions
        self.num_layers = num_layers

        self.embedding = embedding
        self.rnn = nn.GRU(input_size, self.hidden_size,
                          num_layers=num_layers,
                          bidirectional=bidirectional,
                          dropout=dropout)

    def forward(self, input, lengths, h0=None):
        """
        Input:
        input (seq_len, batch): padded sequence tensor
        lengths (1, batch): sequence lengths
        h0 (num_layers*num_directions, batch, hidden_size): initial hidden state
        ---
        Output:
        hn (num_layers*num_directions, batch, hidden_size):
            the hidden state of each layer
        output (seq_len, batch, hidden_size*num_directions): output tensor
        """
        # (seq_len, batch) => (seq_len, batch, input_size)
        embed = self.embedding(input)
        lengths = lengths.data.view(-1).tolist()
        if lengths is not None:
            embed = pack_padded_sequence(embed, lengths) # Compress invalid fill-in values
        output, hn = self.rnn(embed, h0)
        if lengths is not None:
            output = pad_packed_sequence(output)[0] # The compressed sequence is then filled back in for subsequent processing
        return hn, output


class EncoderDecoder(nn.Module):
    # The characteristic dimension of the input vector, input_size= args.vocab_size = 18864
    # The dimension of the hidden layer vector, hidden_size= embedding_size= 256
    # num_layers = 3
    # dropout = args.dropout = 0.2
    # bidirectional = true
    # embedding define here，encoder and decoder share, Map a vocabulary of 18864 terms to a 256-dimensional vector space
    def __init__(self, args, vocab_size):
        super(EncoderDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = args.embedding_size
        self.embedding = nn.Embedding(vocab_size, args.embedding_size,
                                      padding_idx=constants.PAD)
        # generate：hn, output
        # hn (num_layers*num_directions, batch, hidden_size): the hidden state of each layer
        # output (seq_len, batch, hidden_size*num_directions): output tensor
        self.encoder = Encoder(args.embedding_size, args.hidden_size, args.num_layers,
                               args.dropout, args.bidirectional, self.embedding)
        self.num_layers = args.num_layers

    # Convert encoder's output hide layer to decoder's hide layer input
    def encoder_hn2decoder_h0(self, h):
        """
        Input:
        h (num_layers * num_directions, batch, hidden_size): encoder output hn
        ---
        Output:
        h (num_layers, batch, hidden_size * num_directions): decoder input h0
        """
        if self.encoder.num_directions == 2:
            num_layers, batch, hidden_size = h.size(0)//2, h.size(1), h.size(2)
            return h.view(num_layers, 2, batch, hidden_size)\
                    .transpose(1, 2).contiguous()\
                    .view(num_layers, batch, hidden_size * 2)
        else:
            return h

    def forward(self, src, lengths, trg):
        """
        Input:
        src (src_seq_len, batch): source tensor
        lengths (1, batch): source sequence lengths
        trg (trg_seq_len, batch): target tensor, the `seq_len` in trg is not
            necessarily the same as that in src
        ---
        Output:
        output (trg_seq_len, batch, hidden_size)
        """
        encoder_hn, H = self.encoder(src, lengths)
        decoder_h0 = self.encoder_hn2decoder_h0(encoder_hn)
        # Target that removed the EOS put in decoder
        output, decoder_hn = self.decoder(trg[:-1], decoder_h0, H)
        return output
