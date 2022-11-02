import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from positional_encodings import PositionalEncoding1D, PositionalEncoding2D
from einops import rearrange, reduce, repeat




class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout) # encoder_layers = TransformerEncoderLayer(d_model+1, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))  # ADD CLSself.cls_token = nn.Parameter(torch.randn(1, d_model, 1))  # ADD CLS
        # self.encoder = nn.Embedding(ntoken, d_model)
        # self.d_model = d_model
        #self.decoder = nn.Linear(d_model, ntoken)

        # self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        #self.decoder.bias.data.zero_()
        #self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor ) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        # src = self.encoder(src) * math.sqrt(self.d_model)
        # b, _, _ = src.shape
        _, b, _ = src.shape
        tmp = repeat(self.cls_token, '() n e -> n b e', b=b) #tmp = repeat(self.cls_token, '() n e -> e b n', b=b)
        # tmp = tmp.to(self.device)
        # src = src.to(self.device)
        src = torch.cat([tmp, src], dim=0)
        src = self.pos_encoder(src)


        # output = self.transformer_encoder(src, src_mask)
        output = self.transformer_encoder(src )
        #output = self.decoder(output)
        return output





class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.p_enc_1d = PositionalEncoding1D(d_model)


    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        # x = x.view((x.shape[0], x.shape[2], x.shape[1]))
        x = self.p_enc_1d(x)
        # x = x.view((x.shape[0], x.shape[2], x.shape[1]))
        return self.dropout(x)




class PositionalEncoding_2D(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.p_enc_2d = PositionalEncoding2D(d_model)


    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        # x = x.view((x.shape[0], x.shape[2], x.shape[1]))
        x = self.p_enc_2d(x)
        # x = x.view((x.shape[0], x.shape[2], x.shape[1]))
        return self.dropout(x)



class PositionalEncodingYossi(nn.Module):

    def __init__(self, K):

        self.PosEncodingX = nn.Parameter(torch.randn(EmbeddingMaxDim, int(K / 2)))
        self.PosEncodingY = nn.Parameter(torch.randn(EmbeddingMaxDim, int(K / 2)))

    def Prepare2DPosEncoding(self, PosEncodingX, PosEncodingY, RowNo, ColNo):

        #PosEncodingX = PosEncodingX[0:ColNo].unsqueeze(0)     # x=[1,..,20]
        PosEncodingX = PosEncodingX[0:ColNo, :].unsqueeze(1)  # x=[1,..,20]
        #PosEncodingY = PosEncodingY[0:RowNo, :].unsqueeze(1)  # x=[1,..,20]

        for i in range(RowNo):

            CurrentY = PosEncodingY[i, :].unsqueeze(0).unsqueeze(0).repeat(ColNo, 1, 1)

            if i == 0:
                PosEncoding2D = torch.cat((PosEncodingX, CurrentY), 2)
            else:
                CurrentPosEncoding2D =torch.cat((PosEncodingX, CurrentY), 2)

                PosEncoding2D = torch.cat((PosEncoding2D, CurrentPosEncoding2D), 0)

        return PosEncoding2D



class PositionalEncodingOLD(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
