import torch
import torch.nn as nn

class FusionBlock(nn.Module):
    def __init__(self, embed_dim, input_dim=960, num_heads=1, dropout=0.1, norm_layer=nn.LayerNorm):
        super(FusionBlock, self).__init__()

        self.norm_query = norm_layer(embed_dim)
        self.norm_kv = norm_layer(embed_dim)

        self.decoder_align = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=embed_dim, kernel_size=1, bias=False),
            nn.AdaptiveAvgPool2d(output_size=8)
        )

        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.dropout = nn.Dropout(dropout)

    def forward(self, global_feature, decoder_feature):
        query = self.norm_query(global_feature)

        x_dec = self.decoder_align(decoder_feature)
        kv = x_dec.flatten(2).permute(0, 2, 1)
        kv = self.norm_kv(kv)

        attn_output, _ = self.multihead_attn(query, key=kv, value=kv, need_weights=False)
        
        return global_feature + self.dropout(attn_output)


import torch

def main():
    device = torch.device('mps')
    batch_size = 6
    embed_dim = 512
    input_dim = 960
    num_heads = 1

    global_feature = torch.randn(size=[batch_size, 64, embed_dim], device=device)
    decoder_feature = torch.randn(size=[batch_size, input_dim, 256, 256], device=device)

    fusion_block = FusionBlock(embed_dim, input_dim, num_heads).to(device)
    output = fusion_block(global_feature, decoder_feature)

    print(output.shape)
    output = output.mean(dim=1)
    print(output.shape)

if __name__ == '__main__':
    main()