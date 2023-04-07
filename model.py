import torch
import torch.nn as nn
from layers import PositionalEncoder, DecoderLayer, Masking
import math


class Transformer(nn.Module):

    def __init__(self: object, 
                d_model: int,
                n_layers: int,
                n_heads: int,
                d_ff: int,
                max_relative_dist: int,
                max_position: int,
                vocab_size: int,  
                dropout: float,
                norm_eps: float,
                bias = True ) -> None:
        
        super(Transformer, self).__init__()
        self.max_position = max_position
        self.d_model = d_model 

        self.masking_layer = Masking()
        self.embedding_layer = nn.Embedding(vocab_size, d_model)
        self.dropdout_layer = nn.Dropout(dropout)
        self.pe_layer = PositionalEncoder(
            d_model = d_model, 
            max_position = max_position, 
            n_dim = 3, 
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        ) 

        self.decoder_layer = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, max_relative_dist, bias, dropout, norm_eps) for _ in range(n_layers)
        ])

        self.final_layer = nn.Linear(d_model, vocab_size)


    def forward(self: object, x: torch.Tensor) -> torch.Tensor:
        mask = self.masking_layer(x, x.dim()+2)

        # torch.save(x, "./data_gen/model/x.pt")
        # torch.save(mask, "./data_gen/model/mask.pt")

        x = self.embedding_layer(x)

        # torch.save(x, "./data_gen/model/embedding_x.pt")

        if self.max_position:
            x += self.pe_layer.pe[:, :x.shape[-2], :]
        else:
            x *= math.sqrt(self.d_model)
        
        # torch.save(x, "./data_gen/model/pos_x.pt")

        x = self.dropdout_layer(x)

        for idx, layer in enumerate(self.decoder_layer):
            x = layer(x, mask)

        x = self.final_layer(x)

        return x 