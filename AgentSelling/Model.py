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
    """시장 정보 인코딩을 위한 Transformer Encoder"""
    def __init__(
        self,
        ohlcv_dim: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_ff: int,
        max_len: int = 512,
        dropout: float = 0.1,
        pooling: str = "last",  # "last" or "mean"
    ):
        super().__init__()
        self.pooling = pooling
        self.embed = nn.Linear(ohlcv_dim, d_model)
        self.layers = nn.ModuleList(
            [TransformerLayer(d_model, num_heads, dim_ff, dropout, max_len) for _ in range(num_layers)]
        )

    def forward(self, ohlcv_seq: torch.Tensor) -> torch.Tensor:
        # ohlcv_seq: [B,T,ohlcv_dim]
        x = self.embed(ohlcv_seq)  # [B,T,D]
        for layer in self.layers:
            x = layer(x)
        if self.pooling == "mean":
            return x.mean(dim=1)   # [B,D]
        return x[:, -1, :]         # [B,D]

class PortfolioNet(nn.Module):
    def __init__(self, portfolio_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.portfolio_dim = portfolio_dim
        self.network = nn.Sequential(
            nn.Linear(portfolio_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor):
        return self.network(x)
    
class PolicyNet(nn.Module):
    def __init__(self, encode_dim: int, hidden_dim: int, num_actions: int, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(encode_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, market: torch.Tensor, portfolio: torch.Tensor):
        x = torch.cat([market, portfolio], dim=-1)  # 포트폴리오 정보와 시장 정보 결합
        return self.network(x)

if __name__ == "__main__":
    market_encoder = TransformerEncoder(
        ohlcv_dim=5,
        d_model=64,
        num_heads=4,
        num_layers=2,
        dim_ff=256,
        max_len=512,
        dropout=0.1,
        pooling="last",
    )
    balance_net = PortfolioNet(portfolio_dim=2, hidden_dim=32, out_dim=64)
    policy_net = PolicyNet(encode_dim=128, hidden_dim=32, num_actions=3)

    B, T = 8, 64
    dummy_ohlcv = torch.randn(B, T, 5)  # [B=8, T=100, ohlcv_dim=5]
    state = torch.randn(B, 2)      # [B=8, num_assets=3]

    encoded_market = market_encoder(dummy_ohlcv)  # [B, D=64]
    portfolio = balance_net(state)                 # [B, num_actions=3]
    action_logits = policy_net(encoded_market, portfolio)
    print("Encoded Market Shape:", encoded_market.shape)
    print("Portfolio Shape:", portfolio.shape)  
    print("Action Logits Shape:", action_logits.shape)
    print(action_logits)

    dist = torch.distributions.Categorical(logits=action_logits)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    print("Sampled Action:", action)
    print("Log Probability of Action:", log_prob)