import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
# from positional_encodings import PositionalEncoding1D, PositionalEncoding2D
import math


class TransformerModel(nn.Module):

    def __init__(self,  d_model: int, nhead: int, d_hid: int,
                 nlayers: int, seq_len: int, dropout: float = 0.0):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = 'Transformer'

        # self.pos_encoder = PositionalEncoding(d_model, dropout)
        # self.pos_encoder = PositionalEncoding(d_model, dropout, max_len= 2048).to( self.device)  ## WORK
        # self.pos_encoder = torch.nn.Parameter(torch.randn(2048, d_model)).to( self.device) ## WORK better

        # self.pos_encoder = nn.Parameter(torch.randn(256, 20, d_model)).to('cuda')
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first = True ) # encoder_layers = TransformerEncoderLayer(d_model+1, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers).to(self.device )
        self.d_model = d_model
        self.mask = torch.zeros((2048, 2048), requires_grad=False).to('cuda')
        #self.mask = torch.zeros((2*seq_len, 2*seq_len) )
        # self.mask = torch.zeros((256, 256))
        self.mask[0:1024, 0:1024] = float('-inf')
        self.mask[1024:2*1024, 1024:2*1024] = float('-inf')
        self.mask = self.mask.to(self.device)

        # self.init_weights()
    # def init_weights(self) -> None:
    #     initrange = 0.1
    #     # self.cls_token.en
    #     # self.encoder.weight.data.uniform_(-initrange, initrange)
    #     # self.decoder.bias.data.zero_()
    #     # self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor ) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size, embedding]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, embedding]
        """

        # src = src.to(self.device)
        # src[:256, :, :] = self.pos_encoder( src[:256,:,:])
        # src[256:, :, :] = self.pos_encoder(src[256:, :, :])
        # src = self.pos_encoder(src)
        # src[:256, :, :] += self.pos_encoder
        # src[256:, :, :] += self.pos_encoder

        # src = self.pos_encoder(src)
        # src = self.pos_encoder(src) # WORK
        # src = src + self.pos_encoder
        output = self.transformer_encoder(src, self.mask )
        return output



# class PositionalEncoding(nn.Module):
#
#     def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)
#
#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         pe = torch.zeros( 1, max_len,  d_model)
#         pe[0, :, 0::2] = torch.sin(position * div_term)
#         pe[0, :, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x: Tensor) -> Tensor:
#         """
#         Args:
#             x: Tensor, shape [seq_len, batch_size, embedding_dim]
#         """
#         x = x + self.pe[:x.size(0)]
#         return self.dropout(x)
#

#
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model: int, dropout: float = 0.0):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#         self.p_enc_1d = PositionalEncoding2D(d_model)
#
#
#     def forward(self, x: Tensor) -> Tensor:
#         """
#         Args:
#             x: Tensor, shape [seq_len, batch_size, embedding_dim]
#         """
#         # x = x.view((x.shape[0], x.shape[2], x.shape[1]))
#         # x = self.p_enc_1d(x)
#         # x = x.view((x.shape[0], x.shape[2], x.shape[1]))
#         return self.dropout(x)
#




# class PositionalEncoding_2D(nn.Module):
#     def __init__(self, d_model: int, dropout: float = 0.0):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#         self.p_enc_2d = PositionalEncoding2D(d_model)
#
#
#     def forward(self, x: Tensor) -> Tensor:
#         """
#         Args:
#             x: Tensor, shape [seq_len, batch_size, embedding_dim]
#         """
#         # x = x.view((x.shape[0], x.shape[2], x.shape[1]))
#         x = self.p_enc_2d(x)
#         # x = x.view((x.shape[0], x.shape[2], x.shape[1]))
#         return self.dropout(x)


#
# class PositionalEncodingYossi(nn.Module):
#
#     def __init__(self, K):
#
#         self.PosEncodingX = nn.Parameter(torch.randn(EmbeddingMaxDim, int(K / 2)))
#         self.PosEncodingY = nn.Parameter(torch.randn(EmbeddingMaxDim, int(K / 2)))
#
#     def Prepare2DPosEncoding(self, PosEncodingX, PosEncodingY, RowNo, ColNo):
#
#         #PosEncodingX = PosEncodingX[0:ColNo].unsqueeze(0)     # x=[1,..,20]
#         PosEncodingX = PosEncodingX[0:ColNo, :].unsqueeze(1)  # x=[1,..,20]
#         #PosEncodingY = PosEncodingY[0:RowNo, :].unsqueeze(1)  # x=[1,..,20]
#
#         for i in range(RowNo):
#
#             CurrentY = PosEncodingY[i, :].unsqueeze(0).unsqueeze(0).repeat(ColNo, 1, 1)
#
#             if i == 0:
#                 PosEncoding2D = torch.cat((PosEncodingX, CurrentY), 2)
#             else:
#                 CurrentPosEncoding2D =torch.cat((PosEncodingX, CurrentY), 2)
#
#                 PosEncoding2D = torch.cat((PosEncoding2D, CurrentPosEncoding2D), 0)
#
#         return PosEncoding2D
#
#
#
# class PositionalEncodingOLD(nn.Module):
#
#     def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)
#
#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         pe = torch.zeros(max_len, 1, d_model)
#         pe[:, 0, 0::2] = torch.sin(position * div_term)
#         pe[:, 0, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x: Tensor) -> Tensor:
#         """
#         Args:
#             x: Tensor, shape [seq_len, batch_size, embedding_dim]
#         """
#         x = x + self.pe[:x.size(0)]
#         return self.dropout(x)