import math
import torch
import torch.nn as nn

from einops import rearrange
from utils import *

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, padding=1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.append(nn.Sequential(*[
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4, pr=0):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]
        
        hidden_dim = inp * expansion
        
        # apply pruning (if pr == 0, not prune)
        inp = remain_num_after_pr(inp, pr)
        oup = remain_num_after_pr(oup, pr)
        hidden_dim = remain_num_after_pr(hidden_dim, pr)

        self.use_res_connect = self.stride == 1 and inp == oup
        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0., pr=0):
        super().__init__()
        self.ph, self.pw = patch_size

        # apply pruning (if pr == 0, not prune)
        channel = remain_num_after_pr(channel, pr)
        dim = remain_num_after_pr(dim, pr)
        mlp_dim = remain_num_after_pr(mlp_dim, pr)

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)
    
    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Global representations
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x


class MobileViT(nn.Module):
    def __init__(self, image_size, dims, channels, num_classes, expansion=4, kernel_size=3, patch_size=(2, 2), pr=0):
        super().__init__()
        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        L = [2, 4, 3]   # repeated transformer block number
    
        self.conv1 = conv_nxn_bn(3, remain_num_after_pr(channels[0], pr), stride=2)                  # downsampling

        layer1, layer2, layer3, layer4, layer5 = [], [], [], [], []
        
        # output channel: 16(xxs) 32(xs, s)
        layer1.append(MV2Block(channels[0], channels[1], 1, expansion, pr=pr))         # residual connection (only xxs)

        # output channel: 24(xxs) 48(xs), 64(s)
        layer2.append(MV2Block(channels[1], channels[2], 2, expansion, pr=pr))     # downsampling
        layer2.append(MV2Block(channels[2], channels[3], 1, expansion, pr=pr))         # residual connection
        layer2.append(MV2Block(channels[2], channels[3], 1, expansion, pr=pr))         # residual connection

        # output channel: 48(xxs) 64(xs), 96(s)
        layer3.append(MV2Block(channels[3], channels[4], 2, expansion, pr=pr))     # downsampling
        layer3.append(MobileViTBlock(dims[0], L[0], channels[5], kernel_size, patch_size, int(dims[0]*2), pr=pr))

        # output channel: 64(xxs) 80(xs), 128(s)
        layer4.append(MV2Block(channels[5], channels[6], 2, expansion, pr=pr))     # downsampling
        layer4.append(MobileViTBlock(dims[1], L[1], channels[7], kernel_size, patch_size, int(dims[1]*4), pr=pr))

        # output channel: 80(xxs) 96(xs), 160(s)
        layer5.append(MV2Block(channels[7], channels[8], 2, expansion, pr=pr))     # downsampling
        layer5.append(MobileViTBlock(dims[2], L[2], channels[9], kernel_size, patch_size, int(dims[2]*4), pr=pr))

        self.layer1 = nn.Sequential(*layer1)
        self.layer2 = nn.Sequential(*layer2)
        self.layer3 = nn.Sequential(*layer3)
        self.layer4 = nn.Sequential(*layer4)
        self.layer5 = nn.Sequential(*layer5)

        # output channel: 320(xxs) 384(xs), 640(s)
        self.conv1x1 = conv_1x1_bn(remain_num_after_pr(channels[-2], pr), remain_num_after_pr(channels[-1], pr))

        self.pool = nn.AvgPool2d(ih//32, 1)
        self.fc = nn.Linear(remain_num_after_pr(channels[-1], pr), num_classes, bias=False)

    def forward(self, x):
        x = self.conv1(x)   # down-sampling

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.conv1x1(x)

        x = self.pool(x).view(-1, x.shape[1])
        x = self.fc(x)
        return x


def build_MobileVIT(args, config, pr=False):
    num_classes = args.classes
    dims = config.get("dims")
    channels = config.get("channels")
    expansion = config.get("expansion")
    img_size = (args.resize, args.resize)
    if pr:
        return MobileViT(image_size=img_size,
                        dims=dims,
                        channels=channels,
                        num_classes=num_classes,
                        expansion=expansion,
                        pr=args.fprune_rate)

    return MobileViT(image_size=img_size,
                     dims=dims,
                     channels=channels,
                     num_classes=num_classes,
                     expansion=expansion)