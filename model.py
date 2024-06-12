import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn import LayerNorm


##########################################################################
## Layer Norm
def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
# rePatch
# class Attention(nn.Module):
#     def __init__(self, dim, num_heads, bias):
#         super(Attention, self).__init__()
#         self.num_heads = num_heads
#         self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

#         self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
#         self.qkv_dwconv = nn.Conv2d(
#             dim * 3,
#             dim * 3,
#             kernel_size=3,
#             stride=1,
#             padding=1,
#             groups=dim * 3,
#             bias=bias,
#         )
#         self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

#     def forward(self, x):
#         b, c, h, w = x.shape

#         ps = 16

#         qkv = self.qkv_dwconv(self.qkv(x))
#         q, k, v = qkv.chunk(3, dim=1)

#         q = rearrange(
#             q,
#             "b (head c) (ph nh) (pw nw) -> b head (nh nw) (c ph pw)",
#             head=self.num_heads,
#             ph=ps,
#             pw=ps,
#             nh=h // ps,
#             nw=w // ps,
#         )
#         k = rearrange(
#             k,
#             "b (head c) (ph nh) (pw nw) -> b head (nh nw) (c ph pw)",
#             head=self.num_heads,
#             ph=ps,
#             pw=ps,
#             nh=h // ps,
#             nw=w // ps,
#         )
#         v = rearrange(
#             v,
#             "b (head c) (ph nh) (pw nw) -> b head (nh nw) (c ph pw)",
#             head=self.num_heads,
#             ph=ps,
#             pw=ps,
#             nh=h // ps,
#             nw=w // ps,
#         )

#         q = torch.nn.functional.normalize(q, dim=-1)
#         k = torch.nn.functional.normalize(k, dim=-1)

#         attn = (q @ k.transpose(-2, -1)) * self.temperature
#         attn = attn.softmax(dim=-1)

#         out = attn @ v

#         out = rearrange(
#             out,
#             "b head (nh nw) (c ph pw) -> b (head c) (ph nh) (pw nw)",
#             head=self.num_heads,
#             ph=ps,
#             pw=ps,
#             nh=h // ps,
#             nw=w // ps,
#         )

#         out = self.project_out(out)
#         return out


# original
# class Attention(nn.Module):
#     def __init__(self, dim, num_heads, bias):
#         super(Attention, self).__init__()
#         self.num_heads = num_heads
#         self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

#         self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
#         self.qkv_dwconv = nn.Conv2d(
#             dim * 3,
#             dim * 3,
#             kernel_size=3,
#             stride=1,
#             padding=1,
#             groups=dim * 3,
#             bias=bias,
#         )
#         self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

#     def forward(self, x):
#         b, c, h, w = x.shape

#         qkv = self.qkv_dwconv(self.qkv(x))
#         q, k, v = qkv.chunk(3, dim=1)

#         q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
#         k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
#         v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

#         q = torch.nn.functional.normalize(q, dim=-1)
#         k = torch.nn.functional.normalize(k, dim=-1)

#         attn = (q @ k.transpose(-2, -1)) * self.temperature
#         attn = attn.softmax(dim=-1)

#         out = attn @ v

#         out = rearrange(
#             out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
#         )

#         out = self.project_out(out)
#         return out


# Pool
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.qk_pl = nn.MaxPool2d(kernel_size=8, stride=8)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = self.qk_pl(q)
        k = self.qk_pl(k)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v

        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(
            in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def forward(self, x):
        x = self.proj(x)

        return x


###########################  Shared Encoder  ################################
class Shared_Encoder(nn.Module):
    def __init__(
        self,
        in_c=1,
        out_c=32,
        dim=48,
        num_blocks=4,
        heads=8,
        ffn_expansion_factor=2,
        bias=False,
        LayerNorm_type="WithBias",
    ):
        super().__init__()

        self.in_c = in_c
        self.out_c = out_c

        self.patch_embed = OverlapPatchEmbed(in_c, dim)

        self.encoder = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim,
                    num_heads=heads,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks)
            ]
        )

        self.out = nn.Conv2d(dim, self.out_c, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.encoder(x)

        x = self.out(x)

        return x


###########################  Specific Encoder  ################################
class ResBlock(nn.Module):
    def __init__(
        self,
        in_c,
        ffn_expansion_factor=2,
        bias=False,
    ):
        super().__init__()

        dim = in_c * ffn_expansion_factor

        self.conv = nn.Sequential(
            # 3x3
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_c, dim, kernel_size=3, bias=bias),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            # 3x3
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, in_c, kernel_size=3, bias=bias),
            nn.InstanceNorm2d(in_c),
            nn.ReLU(True),
        )

    def forward(self, x):
        return x + self.conv(x)


class Specific_Encoder(nn.Module):
    def __init__(
        self,
        in_c=1,
        out_c=32,
        ffn_expansion_factor=2,
        num_resblocks=9,
        bias=False,
    ):
        super().__init__()

        self.in_c = in_c
        self.out_c = out_c

        dim = 16
        self.embed = nn.Conv2d(self.in_c, dim, kernel_size=1, bias=bias)

        self.feat_extract_7 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(dim, dim * ffn_expansion_factor, kernel_size=7, bias=bias),
            nn.InstanceNorm2d(dim * ffn_expansion_factor),
            nn.ReLU(True),
        )
        dim *= ffn_expansion_factor

        self.feat_extract_5 = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(dim, dim * ffn_expansion_factor, kernel_size=5, bias=bias),
            nn.InstanceNorm2d(dim * ffn_expansion_factor),
            nn.ReLU(True),
        )
        dim *= ffn_expansion_factor

        self.feat_extract_3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim * ffn_expansion_factor, kernel_size=3, bias=bias),
            nn.InstanceNorm2d(dim * ffn_expansion_factor),
            nn.ReLU(True),
        )
        dim *= ffn_expansion_factor

        self.rescnn = nn.Sequential(*[ResBlock(dim) for i in range(num_resblocks)])

        self.out = nn.Conv2d(dim, self.out_c, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.embed(x)
        x = self.feat_extract_7(x)
        x = self.feat_extract_5(x)
        x = self.feat_extract_3(x)

        x = self.rescnn(x)

        x = self.out(x)

        return x


###########################  Decoder  ################################
class Attention_Dec(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.ConvTranspose2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v

        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )

        out = self.project_out(out)
        return out


class FeedForward_Dec(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.ConvTranspose2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class TransformerBlock_Dec(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention_Dec(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward_Dec(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class ResTConvBlock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()

        feat_dim = int(ffn_expansion_factor * dim)
        self.tcnn = nn.Sequential(
            # 1x1
            nn.Conv2d(dim, feat_dim, kernel_size=1, bias=bias),
            nn.InstanceNorm2d(feat_dim),
            nn.ReLU(True),
            # 3x3
            nn.ConvTranspose2d(feat_dim, feat_dim, 3, padding=1, bias=bias),
            nn.InstanceNorm2d(feat_dim),
            nn.ReLU(True),
            # 1x1
            nn.Conv2d(feat_dim, dim, kernel_size=1, bias=bias),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = x + self.tcnn(x)

        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        dim=64,
        num_heads=8,
        ffn_expansion_factor=2,
        bias=False,
        LayerNorm_type="WithBias",
    ):
        super().__init__()

        self.transformer = TransformerBlock_Dec(
            dim=dim,
            num_heads=num_heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
        )

        self.restconv = ResTConvBlock(
            dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias
        )

    def forward(self, x):
        x = x + self.transformer(x)
        x = x + self.restconv(x)

        return x


class TTCDecoder(nn.Module):
    def __init__(
        self,
        in_c=1,
        out_c=1,
        dim=48,
        num_blocks=4,
        heads=8,
        ffn_expansion_factor=2,
        bias=False,
        LayerNorm_type="WithBias",
    ):
        super().__init__()

        self.in_c = in_c
        self.out_c = out_c

        self.embed = nn.Conv2d(self.in_c, dim, kernel_size=1, bias=bias)

        self.decoder = nn.Sequential(
            *[
                DecoderBlock(
                    dim=dim,
                    num_heads=heads,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks)
            ]
        )

        self.out = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(dim, dim * 3, kernel_size=5),
            nn.InstanceNorm2d(dim * 3),
            nn.ReLU(True),
            nn.Conv2d(dim * 3, self.out_c, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.embed(x)
        x = self.decoder(x)
        x = self.out(x)

        return x


class SFModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.comm_enc = Shared_Encoder(in_c=1, out_c=32)
        self.spec_enc = Specific_Encoder(in_c=1, out_c=32)

        self.dec = TTCDecoder(in_c=64, out_c=1)


class SFModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.comm_enc = Shared_Encoder(in_c=1, out_c=32)
        self.spec_enc = Specific_Encoder(in_c=1, out_c=32)

        self.comm_ffm = nn.Conv2d(64, 32, kernel_size=1)
        self.spec_ffm = nn.Conv2d(64, 32, kernel_size=1)

        self.dec = TTCDecoder(in_c=64, out_c=1)
