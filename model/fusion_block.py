import torch
import torch.nn as nn

class FusionBlock(nn.Module):
    def __init__(self, embed_dim, input_dim=960, num_heads=1):
        super(FusionBlock, self).__init__()

        self.global_feature_norm = nn.Sequential(
            nn.BatchNorm1d(num_features=embed_dim),
            nn.ReLU(inplace=True)
        )

        self.decoder_feature_downsample = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=embed_dim, kernel_size=1, bias=False),
            nn.AdaptiveMaxPool2d(output_size=8),
            nn.BatchNorm2d(num_features=embed_dim),
            nn.ReLU(inplace=True)
        )

        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, global_feature, decoder_feature):
        global_feature = global_feature.permute(0, 2, 1)
        query = self.global_feature_norm(global_feature)
        query = query.permute(0, 2, 1)

        decoder_feature_down = self.decoder_feature_downsample(decoder_feature)
        key_value = decoder_feature_down.flatten(2).permute(0, 2, 1)

        attn_output, _ = self.multihead_attn(query, key=key_value, value=key_value, need_weights=False)
        
        return attn_output


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