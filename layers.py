import torch
import torch.nn as nn
import math


class Masking(nn.Module):

    def __init__(self: object) -> None:
        super(Masking, self).__init__()

    
    def forward(self: object, x: torch.Tensor, n_dim: int=4) -> torch.Tensor:
        padding_mask = torch.eq(x, 0).float()
        padding_mask = padding_mask.view(*padding_mask.shape[:-1], *[1 for _ in range(n_dim-2)], padding_mask.shape[-1]).to(x.device)

        look_ahead_mask = torch.triu(torch.ones(x.shape[-1], x.shape[-1]), diagonal=1)
        look_ahead_mask = look_ahead_mask.float().to("cuda:0" if torch.cuda.is_available() else "cpu")

        combine_mask = torch.max(padding_mask, look_ahead_mask)
        return combine_mask

 
class PositionalEncoder(nn.Module):

    def __init__(self: object,
                d_model: int,
                max_position: int,
                n_dim: int,
                device: str) -> None:
        
        super(PositionalEncoder, self).__init__()

        self.d_model = d_model 
        self.max_position = max_position
        self.n_dim =  n_dim

        position = torch.arange(max_position).float().to(device)
        k = torch.div(torch.arange(d_model).float().to(device), 2, rounding_mode='trunc')*2
        wk = 1/torch.pow(10000, k / d_model)
        pe = position.view(-1, 1) @ wk.view(1, -1)
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])

        self.pe = pe.view(*[1]*(n_dim-2), self.max_position, self.d_model)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x *= math.sqrt(self.d_model)
        if self.max_position:
            x += self.pe[:, :x.shape[-2], :]
        return x


class FFNLayer(nn.Module):

    def __init__(self: object,
                d_model: int,
                dff: int,
                bias = True) -> None:
        super(FFNLayer, self).__init__()
        self.d_model = d_model
        self.dff = dff 

        self.ffn_layer = nn.Sequential(
            nn.Linear(d_model, dff, bias=bias),
            nn.ReLU(),
            nn.Linear(dff, d_model, bias=bias)
        )


    def forward(self: object, x: torch.Tensor):
        x = self.ffn_layer(x)
        return x

        
class ScaleDotProductAttention(nn.Module):

    """
    compute scale dot product attention
    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self: object) -> None:
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor=None, e: float=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()
        # print(d_tensor)

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -e)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v
 
        return v, score


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, n_heads: int) -> None:
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_heads
        self.attention = ScaleDotProductAttention().to("cuda:0")
        self.w_q = nn.Linear(d_model, d_model).to("cuda:0")
        self.w_k = nn.Linear(d_model, d_model).to("cuda:0")
        self.w_v = nn.Linear(d_model, d_model).to("cuda:0")
        self.w_concat = nn.Linear(d_model, d_model).to("cuda:0")


    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        return out


    def split(self: object, x: torch.Tensor) -> torch.Tensor:
        """
        split tensor by number of head
        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = x.size()

        d_tensor = d_model // self.n_head
        x = x.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return x


    def concat(self: object, x: torch.Tensor) -> torch.Tensor:
        """
        inverse function of self.split(tensor : torch.Tensor)
        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = x.size()
        d_model = head * d_tensor

        x = x.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return x
 

class DecoderLayer(nn.Module):

    def __init__(self: object,
                d_model: int, 
                n_heads: int,
                dff: int,
                max_relative_dist: int,
                bias = True,
                dropout = 0.1,
                norm_eps = 1e-6) -> None:
        super(DecoderLayer, self).__init__()

        self.d_model = d_model 
        self.n_heads = n_heads
        self.max_relative_dist = max_relative_dist

        self.mha_layer = MultiHeadAttention(d_model, n_heads)
        self.ffn =  FFNLayer(d_model, dff, bias)

        self.norm1_layer = nn.LayerNorm(normalized_shape=d_model, eps=norm_eps)
        self.norm2_layer = nn.LayerNorm(normalized_shape=d_model, eps=norm_eps)

        self.dropout1_layer = nn.Dropout(dropout)
        self.dropout2_layer = nn.Dropout(dropout)

    
    def forward(self:object, x: torch.tensor, x_mask=None) -> torch.Tensor:
        
        att_output = self.mha_layer(x, x, x, mask=x_mask)
        att_output = self.norm1_layer(att_output+x)
        att_output = self.dropout1_layer(att_output)

        ffn_output = self.ffn(att_output)
        ffn_output = self.norm2_layer(ffn_output + att_output)
        ffn_output = self.dropout2_layer(ffn_output)

        return ffn_output