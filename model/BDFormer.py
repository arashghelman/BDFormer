import torch.nn as nn
from model.mt_maxvit import MT_MaxViT
from model.MaxViT.maxxvit_4out import MaxxVitCfg

class BDFormer(nn.Module):  # multi-task netowrk with MaxViT_small_Encoder and MTSwin_Decoder
    def __init__(self, img_size=256, in_channels=1, num_classes=21843, zero_head=False, 
                 window_size=8, has_dropout=False):
        super(BDFormer, self).__init__()
        self.has_dropout = has_dropout
        self.img_size = img_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.window_size = window_size
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

        self.multi_task_MaxViT = MT_MaxViT(img_size=self.img_size,
                                patch_size=4,
                                in_chans=self.in_channels,
                                num_classes=self.num_classes,
                                embed_dim=MaxxVitCfg.embed_dim[0],
                                depths=[2, 2, 2, 2],
                                num_heads=[2, 4, 8, 16],
                                window_size=self.window_size,
                                mlp_ratio=4,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.2,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False,
                                has_dropout=self.has_dropout)

    def forward(self, x):
        if x.size()[1] == 1:
            x = self.conv(x)
        logits = self.multi_task_MaxViT(x)
        return logits