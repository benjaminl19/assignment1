import torch
import math
from torch import nn

from tests import adapters

class Linear(nn.Module):

    def __init__(
            self, 
            in_features: int,
            out_features: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(
            torch.empty(self.out_features, self.in_features, device=device, dtype=dtype)
            )
        
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.W, mean=0.0, std=std, a=-3*std, b=3*std)
    
    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        return torch.einsum('...i,oi->...o', x, self.W)
    
class Embedding(nn.Module):

    def __init__(
            self, 
            num_embeddings: int, 
            embedding_dim: int,
            device: torch.device | None = None,
            dtype:  torch.dtype | None = None
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.M_embed = nn.Parameter(
            torch.empty(self.num_embeddings, embedding_dim, device=device, dtype=dtype)
        )

        nn.init.trunc_normal_(self.M_embed, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(
            self,
            token_ids: torch.Tensor
    ) -> torch.Tensor: 
        
        # flatten to 1-D then select corresponding vector embedding
        seq_embed = self.M_embed.index_select(0, token_ids.reshape(-1))

        # return to original shape
        return seq_embed.reshape(*token_ids.shape, self.M_embed.size(1))
    
class RMSNorm(nn.Module):

    def __init__(
            self, 
            d_model: int,
            eps: float = 1e-5, 
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        # initialize a learnable gain to all ones
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype)) 

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # calculate RMS along d_model dimension (last dim)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

        x = torch.einsum('...d,d->...d', (x / rms), self.gain)

        return x.to(in_dtype)
    
class SwiGLU(nn.Module):

    def __init__(
            self, 
            d_model: int,
            d_ff: int, 
            device: torch.device | None = None, 
            dtype: torch.dtype | None = None,
    ): 
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.W1 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        self.W2 = nn.Parameter(torch.empty(d_model, d_ff, device=device, dtype=dtype))
        self.W3 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        
        silu_w1_x = adapters.run_silu(torch.einsum('fd,...d->...f', self.W1, x))
        w3_x = torch.einsum('fd,...d->...f', self.W3, x)
        silu_w1_x_w3_x = torch.einsum('...f,...f->...f', silu_w1_x, w3_x)
        
        return torch.einsum('df,...f->...d', self.W2, silu_w1_x_w3_x)
        
class RotaryPositionalEmbedding(nn.Module):

    def __init__(
            self,
            theta: float,
            d_k: int,
            max_seq_len: int,
            device: torch.device | None=None
    ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # precompute values of theta_i,k
        exp = torch.arange(0, self.d_k, 2, device=device, dtype=torch.float32)
        period = self.theta ** (-exp/self.d_k)
        index = torch.arange(0, self.max_seq_len, device=device, dtype=torch.float32)
        freq = torch.einsum('i,k->ik', index, period)
        
        # store values of cos, sin in buffer
        self.register_buffer("cos_values", freq.cos(), persistent=False)
        self.register_buffer("sin_values", freq.sin(), persistent=False)

    def forward(
            self,
            x: torch.Tensor,
            token_positions: torch.Tensor
    ) -> torch.Tensor:
        
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # find sin, cos values in buffer
        flat = token_positions.reshape(-1)
        sin = self.sin_values.index_select(0, flat).reshape(*token_positions.shape, self.d_k // 2)
        cos = self.cos_values.index_select(0, flat).reshape(*token_positions.shape, self.d_k // 2)

        # apply rotations to even and odd indices separately
        even_rot = torch.einsum('...sk,...sk->...sk', x[..., 0::2], cos) - torch.einsum('...sk,...sk->...sk', x[..., 1::2], sin)
        odd_rot = torch.einsum('...sk,...sk->...sk', x[..., 0::2], sin) + torch.einsum('...sk,...sk->...sk', x[..., 1::2], cos)
        
        # copy values into output
        output = torch.empty(x.shape, device=x.device, dtype=x.dtype)
        output[..., 0::2] = even_rot
        output[..., 1::2] = odd_rot

        return output.to(in_dtype)

class MultiHeadSelfAttention(nn.Module):

    def __init__(
            self, 
            d_model: int,
            num_heads: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
        ): 
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Parameter(torch.empty(self.d_model, self.d_model, device=device, dtype=dtype))
        self.W_k = nn.Parameter(torch.empty(self.d_model, self.d_model, device=device, dtype=dtype))
        self.W_v = nn.Parameter(torch.empty(self.d_model, self.d_model, device=device, dtype=dtype))
        self.W_o = nn.Parameter(torch.empty(self.d_model, self.d_model, device=device, dtype=dtype))
        
    def forward(
            self,
            x: torch.Tensor,
            max_seq_len: int | None = None,
            theta: float | None = None,
            token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
    
        # one matmul for all 3 projections
        W_qkv = torch.cat([self.W_q, self.W_k, self.W_v], dim=0)
        QKV = torch.einsum('...sd,td->...st', x, W_qkv)
        Q, K, V = torch.split(QKV, (self.d_model, self.d_model, self.d_model), dim=-1)

        # split d_k dim into num_heads, head_dim
        Q = Q.reshape(*Q.shape[:-1], self.num_heads, self.d_k).movedim(-2, -3)
        K = K.reshape(*K.shape[:-1], self.num_heads, self.d_k).movedim(-2, -3)
        V = V.reshape(*V.shape[:-1], self.num_heads, self.d_k).movedim(-2, -3)

        # create causal mask
        q = torch.arange(Q.shape[-2], device=Q.device).unsqueeze(-1) # (seq_len, 1)
        k = torch.arange(K.shape[-2], device=Q.device).unsqueeze(0) # (1, seq_len)
        mask = (q >= k)

        # create token_positions if absent 
        if token_positions is None:
            # creates dimensions [..., seq_len]
            token_positions = torch.arange(Q.shape[-2], device=Q.device, dtype=torch.long)

        if theta is not None:
            # apply rope on each head
            Q = adapters.run_rope(self.d_k, theta, max_seq_len, Q, token_positions)
            K = adapters.run_rope(self.d_k, theta, max_seq_len, K, token_positions)

        attention = adapters.run_scaled_dot_product_attention(Q, K, V, mask)

        # concat heads and project
        attention = attention.movedim(-2, -3).reshape(*x.shape[:-1], self.num_heads * self.d_k)

        return torch.einsum('...sv,dv->...sd', attention, self.W_o)

class TransformerBlock(nn.Module):

    def __init__(
            self, 
            d_model: int, 
            num_heads: int,
            d_ff: int
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.ln1 = RMSNorm(d_model, eps = 1e-5)
        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        self.ln2 = RMSNorm(d_model, eps = 1e-5)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(
            self, 
            x: torch.Tensor,
            max_seq_len: int | None = None,
            theta: float | None = None
    ) -> torch.Tensor: 
        
        y = x + self.attn(self.ln1(x), max_seq_len, theta)
        out = y + self.ffn(self.ln2(y))

        return out

class TransformerLM(nn.Module):

    def __init__(
            self, 
            vocab_size: int,
            d_model: int, 
            num_layers: int,
            num_heads: int,
            d_ff: int,
    ): 
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.embed = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model, eps = 1e-5)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(
            self, 
            x: torch.Tensor,
            context_length: int | None = None,
            rope_theta: float | None = None
    ) -> torch.Tensor:

        # embed input text 
        # (batch, seq_len) -> (batch, seq_len, d_model)
        x = self.embed(x)
        
        # pass through all decoder layers 
        # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        for layer in self.layers:
            x = layer(x, context_length, rope_theta)

        # pass through final rms_norm 
        # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        x = self.ln_final(x)

        # pass through linear layer to get logits 
        # (batch, seq_len, d_model) -> (batch, seq_len, vocab_size)
        x = self.lm_head(x)

        return x
