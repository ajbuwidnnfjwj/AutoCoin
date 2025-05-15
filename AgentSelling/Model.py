import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """토큰 간 상대 거리(i−j)를 임베딩 테이블로 표현"""
    """상대 위치 인코딩 사용"""
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        # 거리 범위: [-(max_len-1) … +(max_len-1)] → 인덱스 [0 … 2*max_len-2]
        self.max_len = max_len
        self.emb = nn.Embedding(2 * max_len - 1, d_model)

    def forward(self, seq_len: int):
        # seq_len × seq_len 거리 매트릭스 생성
        idx = torch.arange(seq_len, device=self.emb.weight.device)
        dist = idx[None, :] - idx[:, None]         # shape: [seq_len, seq_len]
        dist += self.max_len - 1                   # offset to [0…2*max_len-2]
        return self.emb(dist)                      # → [seq_len, seq_len, d_model]

class MultiHeadAttention(nn.Module):
    """Relative Position을 더한 Multi-Head Self-Attention"""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model, self.nh = d_model, num_heads
        self.dh = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, rel_emb: torch.Tensor):
        # x: [B, T, D], rel_emb: [T, T, D]
        B, T, _ = x.size()

        # 1) Q, K, V 생성 및 (B, H, T, dh)로 reshape
        Q = self.W_q(x).view(B, T, self.nh, self.dh).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.nh, self.dh).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.nh, self.dh).transpose(1, 2)

        # 2) 거리 임베딩도 (H, T, T, dh)로 변환
        rel = rel_emb.view(T, T, self.nh, self.dh).permute(2, 0, 1, 3)

        # 3) 콘텐츠 기반 score + 위치 기반 score
        content_scores = Q @ K.transpose(-2, -1)                        # [B,H,T,T]
        position_scores = torch.einsum('bhtd,htsd->bhts', Q, rel)      # [B,H,T,T]

        scores = (content_scores + position_scores) / math.sqrt(self.dh)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # 4) Context & Output
        context = attn @ V                                             # [B,H,T,dh]
        context = context.transpose(1,2).contiguous().view(B, T, self.d_model)
        return self.W_o(context)

class TransformerLayer(nn.Module):
    """Self-Attention + FeedForward + LayerNorm (Relative Positional)"""
    def __init__(self, d_model: int, num_heads: int, dim_ff: int, dropout: float, max_len: int):
        super().__init__()
        self.rel_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.pos_emb = PositionalEncoding(max_len, d_model)

        self.lin1 = nn.Linear(d_model, dim_ff)
        self.lin2 = nn.Linear(dim_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        # x: [B, T, D]
        T = x.size(1)
        rel = self.pos_emb(T)   # [T, T, D]

        # 1) Self-Attention with Relative Pos
        sa = self.rel_attn(x, rel)
        x = self.norm1(x + sa)

        # 2) Feed-Forward
        ff = self.lin2(self.dropout(self.act(self.lin1(x))))
        x = self.norm2(x + ff)
        return x

class TransformerEncoder(nn.Module):
    """여러 레이어를 쌓은 Relative Position Transformer Encoder"""
    def __init__(self,
                 input_dim: int,
                 d_model: int,
                 num_heads: int,
                 num_layers: int,
                 dim_ff: int,
                 num_actions: int,
                 max_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, num_heads, dim_ff, dropout, max_len)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(d_model, num_actions)

    def forward(self, x: torch.Tensor, balance: torch.Tensor):
        # x: [B, T, input_dim], balance: [B, 2]
        balance = balance.unsqueeze(1).repeat(1, x.size(1), 1)
        x = self.embed(torch.cat((x, balance), dim=2))
        for layer in self.layers:
            x = layer(x)
        # 시퀀스 차원 평균 풀링
        x = x.mean(dim=1)
        return self.head(x)         # [B, num_actions]

class MLPHead(nn.Module):
    def __init__(self, d_combined_dim: int, num_actions: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(d_combined_dim, d_combined_dim//2),
            nn.GELU(),
            nn.Linear(d_combined_dim//2, num_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor):
        return self.network(x)

if __name__ == "__main__":
    model = TransformerEncoder(
        input_dim=5,    # 예: [종가, 거래량, RSI, MACD, Signal]
        d_model=64,
        num_heads=4,
        num_layers=2,
        dim_ff=256,
        num_actions=3,  # Buy / Sell / Hold
        max_len=100,
        dropout=0.1
    )
    mlp_head = MLPHead(d_combined_dim=5, num_actions=3)
    dummy = torch.randn(8, 30, 5)  # batch=8, seq_len=30
    out = model(dummy)             # [8, 3]
    balance = torch.randn(1,2)
    out = mlp_head(torch.cat((torch.Tensor(out[-1]), balance.squeeze()), dim=0))
    print(out.shape)  # → torch.Size([8, 3])
    print(out)
