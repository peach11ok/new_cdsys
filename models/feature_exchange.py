import torch.nn as nn
import torch


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)



class ChannelExchange(nn.Module):

    def __init__(self, p=2):
        super(ChannelExchange, self).__init__()
        self.p = p
        self.sam = SpatialAttention(kernel_size=3)
    def forward(self, x1, x2):
        print(f'----------------{type(x1)} and {x1.shape}-----------------')
        attention_map1 = self.sam(x1)
        attention_map2 = self.sam(x2)
        print(f'-------{type(attention_map1)} and attention_map1.shape is {attention_map1.shape}-----')

        avg_weight1 = torch.mean(attention_map1)
        avg_weight2 = torch.mean(attention_map2)
        lower_weight1 = attention_map1 < avg_weight1
        lower_weight2 = attention_map2 < avg_weight2
        print(f'*******{type(lower_weight1)} and lower_weight1.shape is {lower_weight1.shape}********')
        exchange_map1 = attention_map1.clone()
        exchange_map2 = attention_map2.clone()

        # =====
        exchange_map1[lower_weight1] = attention_map2[lower_weight1]
        exchange_map2[lower_weight2] = attention_map1[lower_weight2]
        out_x1 = x1 * (exchange_map1 + attention_map1)
        out_x2 = x2 * (exchange_map2 + attention_map2)
        return out_x1, out_x2



class SpatialExchange(nn.Module):


    def __init__(self, p=2):
        super(SpatialExchange, self).__init__()
        self.p = p
        self.sam = SpatialAttention(kernel_size=3)
    def forward(self, x1, x2):
        attention_map1 = self.sam(x1)
        attention_map2 = self.sam(x2)

        avg_weight1 = torch.mean(attention_map1)
        avg_weight2 = torch.mean(attention_map2)
        lower_weight1 = attention_map1 < avg_weight1
        lower_weight2 = attention_map2 < avg_weight2
        exchange_map1 = attention_map1.clone()
        exchange_map2 = attention_map2.clone()

        # =====
        exchange_map1[lower_weight1] = attention_map2[lower_weight1]
        exchange_map2[lower_weight2] = attention_map1[lower_weight2]
        out_x1 = x1 * (exchange_map1 + attention_map1)
        out_x2 = x2 * (exchange_map2 + attention_map2)
        return out_x1, out_x2
