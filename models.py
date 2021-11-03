import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
import constants

class Encoder(nn.Module):
    # 输入向量的特征维度 input_size = args.vocab_size = 18864
    # 隐藏层 向量维度 hidden_size = args.embeddingsize= 256
    # num_layers = args.hidden_size = 3
    # embedding (vocab_size, input_size): pretrained embedding 提前训练好了的
    def __init__(self, input_size, hidden_size, num_layers, dropout,
                       bidirectional, embedding):

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
            embed = pack_padded_sequence(embed, lengths) # 压缩掉无效的填充值
        output, hn = self.rnn(embed, h0)
        if lengths is not None:
            output = pad_packed_sequence(output)[0] # 压紧的序列再填充回来，便于进行后续的处理
        return hn, output


class EncoderDecoder(nn.Module):
    # 输入向量的特征维度 input_size= args.vocab_size = 18864
    # 隐藏层 向量维度 hidden_size= embedding_size= 256
    # num_layers = 3
    # dropout = args.dropout = 0.2
    # bidirectional = true
    # embedding 在此处定义，encoder and decoder共享，是将18864的词汇表映射到一个256维度的向量空间
    def __init__(self, vocab_size, embedding_size,
                       hidden_size, num_layers, dropout, bidirectional):
        super(EncoderDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size, embedding_size,
                                      padding_idx=constants.PAD)
        # 生成：hn, output
        # hn (num_layers*num_directions, batch, hidden_size): the hidden state of each layer
        # output (seq_len, batch, hidden_size*num_directions): output tensor
        self.encoder = Encoder(embedding_size, hidden_size, num_layers,
                               dropout, bidirectional, self.embedding)
        self.num_layers = num_layers



    # 将encoder的输出隐藏层转化为decoder的隐藏层输入
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
        ## target去除EOS行后调入decoder
        output, decoder_hn = self.decoder(trg[:-1], decoder_h0, H)
        ''' test '''
        return output
