import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import copy
from .feature_exchange import SpatialExchange, ChannelExchange
# from new_feature_exchange import SpatialExchange, ChannelExchange
import math
from .Deep_guide_Shallow_BFDFF import bfdff
from .cross_attention_only_bifea_not_adjacency import CA_BiFeature

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    # B为一个batch中的数量
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)  # 如果换成384输入，这里无法被整除
    # shape为（B,划分后的高，window_size，划分后的宽，window_size,C）
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    # window的shape为（Bx划分后的高x划分后的宽,window_size,window_size,C）
    # 前面的Bx划分后的高x划分后的宽=一个batch中窗口的数量
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    # 因为可以看到，前面的window实际上是一个batch中所有窗口的数量
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    就是每个窗口计算自己的注意力，一共需要计算的窗口数量为输入特征的高/window_size * 输入特征的宽/window_size。

    在实际执行过程中，W-SMA和SW-SMA过程是一样的，即都是按照同样的窗口数，唯一不同的时候SW-SMA需要对结果进行mask。
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        # 输入通道维数
        self.window_size = window_size  # Wh, Ww
        # 每个window包含patch的数量，HxW个
        self.num_heads = num_heads
        # 注意力的头数
        head_dim = dim // num_heads   # 每一个attention头处理的维度数
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        '''
        torch.meshgrid([coords_h,coords_w])  生成网格
        其中第一个输出张量填充第一个输入张量的元素，同一行的每个元素相等，不同行之间的元素分别对应输入张量中的元素
        第二个输出张量填充第二个输入张量的元素，列中的每个元素相等，不同列之间的元素分别对应输入张量中的元素
        '''
        coords_flatten = torch.flatten(coords, 1)  # shape为：2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        """
        coords_flatten[:,:,None] -->将coords_flatten按照行列排序的顺序转换为列，shape为：2,49,1
        coords_flatten[:, None, :] -->将shape转变为2,1,49
        relative_coords的shape为：2,49,49
        是因为一个window中的patch计算注意力，一共需要计算M^2 * M^2次
        """
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        """
        permute相当于reshape，contiguous则是深拷贝，即对新的relative_coords进行修改并不会影响原来的relative_coords
        并将relative_coords的shape变为：49,49,2
        即：将原先的两个49x49按照行与行之间的对应关系，按照顺序，每次取对应行中的单个值组成一个1x2的矩阵。
        """
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        """
        relative_coords[:,:,0]的shape为49x49，即将49x49x1
        relative_coords[:, :, 1]的shape同样为49x49
        这两个中的每个值都+6是为了让索引值从0开始
        """
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        # 那这一步是为了干什么呢
        relative_position_index = relative_coords.sum(-1)  # shape为Wh*Ww, Wh*Ww
        # 这里是按照最后一个维度求和
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        # 截断正态分布
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        # 这里的x.shape其实指的是用window对特征图进行划分后得到的，shape为：窗口数量x窗口的宽x窗口的高x通道维度
        # 因此这里的N实际上就是窗口的大小，如7x7=49;B实际上就是窗口的数量
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # 一次性获得三个矩阵，其shape为[三个矩阵，窗口数量，heads，一个窗口中需要计算自注意的次数，每个头计算的维度数量]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # [窗口数量，heads，一个窗口中需要计算自注意的次数，每个头计算的维度数量]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))   # shape为[256,3,49,49]

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        # Wh*Ww,Wh*Ww,nH   shape[计算自注意的次数,计算自注意的次数,heads]
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # 进入到SW-MSA阶段，需要进行mask
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            # 因为现在是W-MSA，所以不需要做mask
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
        # x的shape为[窗口数量，窗口中一个元素需要做自注意力的次数，维度]

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        # 参数和swin transformer原来的有所差异，少了一个fused_window_process=False
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C= x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        # 但是先做W-MSA   后SW-MSA
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            # torch.roll(input,shifts,dim=None).其中的shifts表示的是张量元素移位的位数。并且dims的大小必须和shifts一致
            # 这里相当于在1维移动-self.shift_size位，在2维移动-self.shift_size位。其中负数表示向上和向左移动
            # 这里和源代码比较，少了一个if not self.fused_window_process的代码块，这个代码块中是关于窗口划分的代码
        else:
            shifted_x = x
            # 同样这里少了一个和window划分的一行代码

        # 先是没有偏移的
        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        # nW*B（窗口个数xbatch）, window_size, window_size, C——————对图片进行划分
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        # 其实就是在每次attention之后将特征还原到原先的大小

        # reverse cyclic shift--因为之前实现了窗口的偏移，attention之后需要进行还原
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    '''
    将图像转换为patches_resolution的图像，每个patch视为一个token，该token的特征是该patch RGB值的展开，token_feature=48
    '''
    def __init__(self, img_size=256, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)  # 变成(256 ,256)
        # to_2tuple(parameter)，将parameter转换为长度为2的元组，若parameter长度已经为2，则不作任何处理
        patch_size = to_2tuple(patch_size)  # 变成（4,4）
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]  # 整数除法
        # patch_resolution也就是一张图片包含的patch数量，如56x56
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        # linearProjection将维度转换成C，C=dim=96 Swin-T Swin-S的配置,

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        # B Ph*Pw C，其中的Ph*Pw为patch_resolution的大小，即一张图片高宽可以被划分为多少个patch
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        # 整张图片被patch分割出来的高宽
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    """
    patch_merging实际上就是CNN中池化操作。让分辨率下降，通道数倍增。即就是一个下采样
    在这里，转换的是patches的分辨率，而不是图像本身的分辨率
    """
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim   # 输入通道数
        # nn.Linear(输入二维张量的大小, 输出二维张量大小, bias=False)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        # 因为每层的输入，其高宽都是一个偶数
        x = x.view(B, H, W, C)
        # 相当于reshape


        # 这一块实际上就是做Merging,即下采样
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        # x[B;H从0开始，步长为2取值;W从0开始，步长为2取值;C]
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        # x[B;H从1开始，步长为2;W从0开始，步长为2;C]
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        # x[B;H从0开始，步长为2;W从1开始，步长为2;C]
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        # x[B;H从1开始，步长为2;W从1开始，步长为2;C]
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
        # x.view(B,-1,4 * C)代表将x中的H和W展平，即H/2*W/2
        x = self.norm(x)
        x = self.reduction(x)
        # 通过全连接层将4*C 变成2*C
        return x
        # x.shape = B,H/2*W/2,2C

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        # nn.Identity() ------> 不改变输入
        self.norm = norm_layer(dim // dim_scale) # norm_layer(512)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        # x.shape = (B, 2H, 2W, C//4)
        x = x.view(B, -1, C // 4)
        # x.shape = (B,4HW, C//4)
        x = self.norm(x)
        # x。shape = (B,4HW, C//4)
        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm, patchsize=4):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, patchsize * patchsize * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)

        return x


class FinalPatchExpand_X4_1(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=1, norm_layer=nn.LayerNorm, patchsize=1):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        # self.expand = nn.Linear(dim, patchsize*patchsize*dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        # x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution      # 56x56
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,  # 相邻两个trans blocks, 一个shift，一个不用
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        # print(f'5-------{x.shape}')
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        # print(f'6-------{x.shape}')
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class BasicLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class SwinTransEncoder(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=2,
                 embed_dim=192, depths=[2, 2, 18, 2], depths_decoder=[4, 4, 4, 4], num_heads=[6, 12, 24, 48],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", **kwargs):
        # 参数传递过来后，实际上为：
        """
        img_size=224, patch_size=4, in_chans=3, num_classes=2,
        embed_dim=128, depths=[2, 2, 18, 2], depths_decoder=[4, 4, 4, 4],
        num_heads=[4, 8, 16, 32],
        window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
        norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
        use_checkpoint=False, final_upsample="expand_first"
        其中会在patchEmbedding之后加上一个normalization
        """
        super().__init__()

        print(
            "SwinTransformerSys expand initial----depths:{};depths_decoder:{};drop_path_rate:{};num_classes:{}".format(
                depths,
                depths_decoder, drop_path_rate, num_classes))

        self.num_classes = num_classes   # 2
        self.num_layers = len(depths)    # 4
        self.embed_dim = embed_dim       # 128
        self.ape = ape                   # 此处为添加相对位置编码
        self.patch_norm = patch_norm     # 在patchEmbedding之后加上normalization
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))   # 128 x 2^3 = 1024，即最后一层的输出的特征维度
        self.num_features_up = int(embed_dim * 2)            # 如果是从x2的角度来看，或许是双时态特征？
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample      # 这个final_upsample是干什么的目前还不知道

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.deal = nn.ModuleList()
        self.deal.append(nn.LayerNorm(128))
        self.deal.append(nn.LayerNorm(256))
        self.deal.append(nn.LayerNorm(512))
        self.deal.append(nn.LayerNorm(1024))

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build encoder and bottleneck layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)
        self.EX = ChannelExchange()
        self.SE = SpatialExchange()
        self.norm = norm_layer(self.num_features)
        self.fusion = nn.Conv2d(1024*2, 1024, 3, 1, 1)

    # Encoder and Bottleneck
    def transpose(self, x):
        B, HW, C = x.size()
        H = int(math.sqrt(HW))     # 求平方根
        x = x.transpose(1, 2)      # （B,HW,C） -> (B, C, HW)
        x = x.view(B, C, H, H)     # (B, C, HW) -> (B, C, H, W)
        return x

    def transpose_verse(self, x):
        B, C, H, W = x.size()
        x = x.view(B, C, -1)      # (B, C, H, W) -> (B, C, HW)
        x = x.transpose(1, 2)     # (B, C, HW) -> (B, HW, C)
        return x

    def forward(self, x1, x2):
        # input [1,3,224,224]
        x1 = self.patch_embed(x1)  # [1,56x56,96(embedding dim)]
        # print(f'1-------{x1.shape}')
        x2 = self.patch_embed(x2)  # [1,56x56,96(embedding dim)]
        if self.ape:
            x1 = x1 + self.absolute_pos_embed
            x2 = x2 + self.absolute_pos_embed
        x1 = self.pos_drop(x1)
        x2 = self.pos_drop(x2)
        x1_downsample = []
        x2_downsample = []
        x_diff = []

        for inx, layer in enumerate(self.layers):
            if inx != 3:
                x1_downsample.append(self.deal[inx](x1))
                x2_downsample.append(self.deal[inx](x2))
                if inx == 2:
                    x1_clone = x1.clone()
                    x2_clone = x2.clone()
                    x1, x2 = self.EX(self.transpose(x1), self.transpose(x2))
                    x1 = self.transpose_verse(x1) + x1_clone
                    x2 = self.transpose_verse(x2) + x2_clone
                x1 = layer(x1)  # ??norm(  mlp(layer_norm(self-attention(x)))
                x2 = layer(x2)
                x_diff.append(x1 - x2)
            else:
                x1_clone = x1.clone()
                x2_clone = x2.clone()
                x1_downsample.append(self.deal[inx](x1))  # self.deal[inx](x))  #self.deal[inx]
                x2_downsample.append(self.deal[inx](x2))
                x1, x2 = self.SE(self.transpose(x1), self.transpose(x2))
                x1 = self.transpose_verse(x1) + x1_clone
                x2 = self.transpose_verse(x2) + x2_clone
                x1 = layer(x1)
                x2 = layer(x2)
                x_diff.append(x1 - x2)
                x_mid = self.transpose_verse(self.fusion(self.transpose(torch.cat([x1, x2], dim=2))))
        x_mid = self.norm(x_mid)  # B L C --1 49 768]

        return x_mid, x1_downsample, x2_downsample, x_diff


# TODO finish this part to transfer trans 2 cnn
class PatchUnembed(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, final_dim=64, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale

        self.final_dim = final_dim

        self.expand = nn.Linear(dim, 16 * final_dim, bias=False)
        self.output_dim = dim // self.dim_scale ** 2

        self.norm = norm_layer(self.final_dim)

        # self.output = nn.Conv2d(in_channels=self.output_dim,out_channels=self.final_dim,kernel_size=1,bias=False)

    def forward(self, x):
        """
        x: B, H*W, C
        -> B, C/16, H*4, W*4
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = self.expand(x)
        C = 16 * self.final_dim
        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.final_dim)
        x = self.norm(x)

        x = x.view(B, 4 * H, 4 * W, -1)
        x = x.permute(0, 3, 1, 2)  # B,C,H,W
        # x = self.output(x)

        return x


class SwinTransDecoder(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=2,
                 embed_dim=128, depths=[4, 4, 4, 4], depths_decoder=[2, 2, 2, 2], num_heads=[4, 8, 16, 32],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample
        self.patches_resolution = [img_size // patch_size, img_size // patch_size]
        self.patch_size = patch_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.concat_linear0 = nn.Linear(1024, 1024)

        self.norm = nn.ModuleList()
        self.norm.append(nn.LayerNorm(512))
        self.norm.append(nn.LayerNorm(256))
        self.norm.append(nn.LayerNorm(128))
        self.norm.append(nn.LayerNorm(128))

        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                      int(embed_dim * 2 ** (
                                                  self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer == 0:
                layer_up = PatchExpand(
                    input_resolution=(self.patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                      self.patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                         input_resolution=(
                                         self.patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                         self.patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                                         depth=depths[(self.num_layers - 1 - i_layer)],
                                         num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                                         window_size=window_size,
                                         mlp_ratio=self.mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop_rate, attn_drop=attn_drop_rate,
                                         drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                                             depths[:(self.num_layers - 1 - i_layer) + 1])],
                                         norm_layer=norm_layer,
                                         upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                                         use_checkpoint=use_checkpoint)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm_up = norm_layer(self.embed_dim)
        if self.final_upsample == "expand_first":
            # print("---final upsample expand_first---")
            self.up = FinalPatchExpand_X4(input_resolution=(img_size // patch_size, img_size // patch_size),
                                          dim_scale=4, dim=embed_dim, patchsize=patch_size)
            self.up_0 = FinalPatchExpand_X4_1(input_resolution=(56, 56), dim_scale=1, dim=128, patchsize=1)
            self.up_1 = FinalPatchExpand_X4_1(input_resolution=(28, 28), dim_scale=1, dim=256, patchsize=1)
            self.up_2 = FinalPatchExpand_X4_1(input_resolution=(14, 14), dim_scale=1, dim=512, patchsize=1)
            self.output = nn.Conv2d(in_channels=embed_dim, out_channels=self.num_classes, kernel_size=1, bias=False)
            self.output_0 = nn.Conv2d(in_channels=128, out_channels=self.num_classes, kernel_size=1, bias=False)
            self.output_1 = nn.Conv2d(in_channels=256, out_channels=self.num_classes, kernel_size=1, bias=False)
            self.output_2 = nn.Conv2d(in_channels=512, out_channels=self.num_classes, kernel_size=1, bias=False)

        self.conv = nn.ModuleList()
        self.conv.append(nn.Conv2d(1024 * 2, 1024, 3, 1, 1))
        self.conv.append(nn.Conv2d(512 * 2, 512, 3, 1, 1))
        self.conv.append(nn.Conv2d(256 * 2, 256, 3, 1, 1))
        self.conv.append(nn.Conv2d(128 * 2, 128, 3, 1, 1))

        self.norm_bn = nn.ModuleList()
        self.norm_bn.append(nn.BatchNorm2d(2048))
        self.norm_bn.append(nn.BatchNorm2d(1024))
        self.norm_bn.append(nn.BatchNorm2d(512))
        self.norm_bn.append(nn.BatchNorm2d(256))

    def transpose(self, x):
        B, HW, C = x.size()
        H = int(math.sqrt(HW))
        x = x.transpose(1, 2)
        x = x.view(B, C, H, H)
        return x

    def transpose_verse(self, x):
        B, C, H, W = x.size()
        x = x.view(B, C, -1)
        x = x.transpose(1, 2)
        return x

    def forward_up_features(self, o1, o2, o3, o4):  # 1/4,1/8,1/16,1/32,     1/32
        x_upsample = []
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = self.transpose_verse(o1)
                # pyramid with PAM
                x = self.concat_linear0(x)  # C,C  1024     nn.Linear(1024, 1024)
                y1 = layer_up(x)  # C/2  UP SAMPLE  512
                y2 = y1  # B,HW,C/2     512
                x = torch.cat([y1, y2], dim=2)  # C    B,HW,C     1024
                x_upsample.append(self.norm[0](y1))  # B,HW,C/2
            else:
                x = self.transpose(x)  # 1024
                if inx == 1:
                    hidden = torch.cat([o2, o2], dim=1) + x
                    x = self.conv[inx](hidden)
                if inx == 2:
                    hidden = torch.cat([o3, o3], dim=1) + x
                    x = self.conv[inx](hidden)
                if inx == 3:
                    hidden = torch.cat([o4, o4], dim=1) + x
                    x = self.conv[inx](hidden)
                x = self.transpose_verse(x)  # B,HW,C
                x = self.concat_back_dim[inx](x)  ######
                y1 = layer_up(x)  # layer up 初始层有norm,up  norm,norm,up
                y2 = y1
                x = torch.cat([y1, y2], dim=2)  # C1024
                norm = self.norm[inx]
                x_upsample.append((norm(y1)))
            # my second module=====

        x = self.norm_up(y1)  # B L C   最终预测结果

        return x, x_upsample

    def up_x4(self, x, pz):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"

        if self.final_upsample == "expand_first":
            x = self.up(x)
            x = x.view(B, pz * H, pz * W, -1)
            x = x.permute(0, 3, 1, 2)  # B,C,H,W
            x = self.output(x)

        return x

    def up_x4_1(self, x, pz):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"

        if self.final_upsample == "expand_first":
            x = self.up_0(x)
            x = x.view(B, pz * H, pz * W, -1)
            x = x.permute(0, 3, 1, 2)  # B,C,H,W
            x = self.output_0(x)

        return x

    def up_x8(self, x, pz):
        H, W = (28, 28)
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"

        if self.final_upsample == "expand_first":
            # x = self.up(x,patchsize=pz)
            x = self.up_1(x)
            x = x.view(B, pz * H, pz * W, -1)
            x = x.permute(0, 3, 1, 2)  # B,C,H,W
            x = self.output_1(x)

        return x

    def up_x16(self, x, pz):
        H, W = (14, 14)
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"

        if self.final_upsample == "expand_first":
            # x = self.up(x,patchsize=pz)
            x = self.up_2(x)
            x = x.view(B, pz * H, pz * W, -1)
            x = x.permute(0, 3, 1, 2)  # B,C,H,W
            x = self.output_2(x)

        return x

    def forward(self, o1, o2, o3, o4):
        x, x_upsample = self.forward_up_features(o1, o2, o3, o4)

        x_p = self.up_x4(x, self.patch_size)
        x_pre2 = self.up_x4_1(x_upsample[2], 1)
        x_pre3 = self.up_x8(x_upsample[1], 1)
        x_pre4 = self.up_x16(x_upsample[0], 1)

        return x_p, x_pre2, x_pre3, x_pre4


class encoder1(nn.Module):
    def __init__(self):
        super(encoder1, self).__init__()
        self.encoder1 = SwinTransEncoder(img_size=224, patch_size=4, in_chans=3, num_classes=2, embed_dim=128,
                                         depths=[2, 2, 18, 2], depths_decoder=[4, 4, 4, 4], num_heads=[4, 8, 16, 32],
                                         window_size=7)
        self.pretrained_path = '/mnt/zt/FIFFST/pretrained/swin_base_patch4_window7_224_22kto1k.pth'
        self.load_from()

    def load_from(self):
        pretrained_path = self.pretrained_path
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model" not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.encoder1.load_state_dict(pretrained_dict, strict=False)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.encoder1.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3 - int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k: v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    # print(1)
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, v.shape, model_dict[k].shape))
                        del full_dict[k]

            msg = self.encoder1.load_state_dict(full_dict, strict=False)
        else:
            print("none pretrain")

    def forward(self, img1, img2):
        x, y1, y2, diff = self.encoder1(img1, img2)
        return x, y1, y2, diff


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.encoder1 = encoder1()
        self.CA_BiFea = CA_BiFeature()
        self.decoder = SwinTransDecoder()

    def forward(self, img1, img2):
        x, x_downsample1, x_downsample2, x_diff= self.encoder1(img1, img2)
        ca1, ca2, ca3, ca4 = self.CA_BiFea(x_diff, x_downsample1, x_downsample2)
        x_p, x_2, x_3, x_4 = self.decoder(ca1, ca2, ca3, ca4)

        return x_p, x_2, x_3, x_4
