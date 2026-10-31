import torch.nn as nn
import torch

class ExampleBlock(nn.Module):
    def __init__(self,in_size,out_size,):
        super().__init__()

        self.norm = nn.LayerNorm(in_size)
        self.mlp = nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.ReLU(),
            nn.Linear(out_size, out_size),
            nn.SiLU()
        )

    def forward(self, x):
        # x is (B,T,C)
        return self.mlp(self.norm(x))


class ExampleNet(nn.Module):
    def __init__(self, size):
        super().__init__()

        self.block1 = ExampleBlock(size, size)

        self.att1 = nn.MultiheadAttention(size,num_heads=1, batch_first=True)

        self.block2 = ExampleBlock(size, size*4)

        self.block3 = ExampleBlock(size*4, size)

        self.out = nn.Linear(size, 1)
    
    def forward(self, x):

        h = self.block1(x)

        h, _ = self.att1(h,h,h)

        x = x + h

        h = self.block2(x)

        h = self.block3(h)

        x = x + h

        return self.out(x)

