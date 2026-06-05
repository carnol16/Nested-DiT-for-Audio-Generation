import torch.nn as nn
import torch


class CrossAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.multihead = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, context):
        Q = x
        K = context
        V = context

        attn_out, weights = self.multihead(Q,K,V)

        normed_output = self.norm(attn_out)


        return normed_output, weights



if __name__ == "__main__":
    ca = CrossAttention(hidden_dim=384, num_heads=4)
    x = torch.randn(2, 348, 384)
    ctx = torch.randn(2, 58, 384)
    out, weights = ca(x, ctx)
    print(out.shape)      # torch.Size([2, 348, 384])
    print(weights.shape)  # torch.Size([2, 348, 58])
