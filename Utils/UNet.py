
# UNet 
# Inconv - Encoder (DownBlocks) - Bottleneck - Decoder (Upblocks) - OutNorm - OutAct - OutConv
# Reminder: The first downblock in the UNet does NOT double channels
# Reminder: It looks like there is a skip shape missmatch but the upsample in the upblocks converts to ch_out so the skip that goes into an UpBlock must have ch_out channels

import torch.nn as nn
import torch
import torch.fx




class DepthwiseSeparableConv(nn.Module):
    def __init__(
        self,
        activation_fn,
        ch_in: int,
        ch_out: int,
        stride: int = 1,
        kernel_size: int = 3
    ):
        super().__init__()

        padding = kernel_size // 2

        # Depthwise Convolution (groups = in_channels)
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                in_channels=ch_in,
                out_channels=ch_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=ch_in, # !!!
                bias=False
            ),
            nn.GroupNorm(num_groups=min(32,ch_in),num_channels=ch_in),
            activation_fn()
        )

        # Pointwise Convolution (1x1 Conv)
        self.pointwise = nn.Conv2d(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class FlashMHA(nn.Module):
    """Flash-Multihead-Selfattention for (B,T,C) inputs, disable_flash can be used to compare to normal att for testing purposes."""
    def __init__(self, dim: int, num_heads: int, dropout: float, disable_flash: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.disable_flash = disable_flash

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.dropout_p = dropout

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        if self.disable_flash:
            scale = self.head_dim ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)

            if self.training and self.dropout_p > 0:
                attn = nn.functional.dropout(attn, p=self.dropout_p)

            out = attn @ v
        else:
            # this also handles scaling
            out = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p if self.training else 0.0)

        out = out.transpose(1, 2).contiguous().view(B, N, C)
        return self.dropout(out)  # B, N, C

class AttentionBlock(nn.Module):
    """Flash-MultiHeadSelfAttention for B,C,H,W inputs with residual scaling"""
    def __init__(self, config, dim: int):
        super().__init__()

        assert dim % 32 == 0, f"Dim must be divisible by 32. dim :{dim}"
        self.n_heads = min(8, dim // 32)
        
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)

        self.norm = nn.LayerNorm(dim)
        self.att = FlashMHA(dim=dim, num_heads=self.n_heads, dropout=config.dropout)
        self.proj = nn.Linear(dim,dim)

        # Zero-init the projection layer so att starts as identity function
        if config.do_zero_init:
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        torch._assert(x.dim() == 4, f"x must be of shape (B,C,H,W) but got {x.shape}")
        B,C,H,W = x.shape

        h = x.permute(0,-2,-1,1) # -> B,H,W,C
        h = h.view(B,H*W,C) # -> B,T,C

        h = self.norm(h)

        h_att  = self.att(h)
        h_att = self.proj(h_att)
        # Scale attention output for smaller gradients and more stable training
        h_att = h_att * self.residual_scale

        h_att = h_att.permute(0, 2, 1).view(B, C, H, W)

        return h_att + x # residual connection with scaled attention output

class ResBlock(nn.Module):
    """A single residual block: save residual connection - Norm-Act-Conv-Dropout - AdaGN projection of embedding - Norm-Act-Conv - +Residual Connection """
    def __init__(self, config, ch_in: int, ch_out: int, depthwise_separable: bool = False):
        super().__init__()

        self.dropout = nn.Dropout(config.dropout)

        # First path: norm -> act -> conv
        self.norm1 = nn.GroupNorm(num_groups=min(32,ch_in),num_channels=ch_in) 
        self.act1 = config.activation_function()
        self.conv1 = DepthwiseSeparableConv(activation_fn=config.activation_function, ch_in=ch_in, ch_out=ch_out) if depthwise_separable else nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1, bias=False)

        # AdaGN: project embedding vector to create scale & shift
        mlp_layer = nn.Sequential(nn.Linear(config.embvector_dim, config.embvector_dim), config.activation_function())
        mlp = [mlp_layer for _ in range(config.embedding_vector_mlp_layers-1)]
        self.embedding_mlp = nn.Sequential(*mlp, 
                                           nn.Linear(config.embvector_dim, 2*ch_out),
                                           config.activation_function())

        # Second path: norm (AdaGN) -> act -> conv
        self.norm2 = nn.GroupNorm(num_groups=min(32,ch_out),num_channels=ch_out) # this will be the norm used in AdaGN
        self.act2 = config.activation_function()
        self.conv2 = DepthwiseSeparableConv(activation_fn=config.activation_function, ch_in=ch_out, ch_out=ch_out) if depthwise_separable else nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1)

        # Residual projection path: norm -> conv 
        if ch_in != ch_out:
            self.norm_res = nn.GroupNorm(num_groups=min(32,ch_in),num_channels=ch_in)
            self.res_proj = nn.Conv2d(ch_in, ch_out, kernel_size=1) # projects the residual connection to ch_out channels
        else:
            self.norm_res = nn.Identity()
            self.res_proj = nn.Identity()

        if config.do_zero_init:
            if depthwise_separable:
                # Target the pointwise layer inside DepthwiseSeparableConv
                target_layer = self.conv2.pointwise
            else:
                target_layer = self.conv2

            # Initialize the final convolution to zero. This ensures the block starts as an identity function.
            nn.init.zeros_(target_layer.weight)
            if target_layer.bias is not None:
                nn.init.zeros_(target_layer.bias)

    def forward(
        self,
        x: torch.Tensor,
        embedding_vector: torch.Tensor,
        return_bottleneck: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # x is (B,C,H,W)
        # embedding vector is (B, embvector_dim)

        # First path: norm -> act -> conv -> dropout
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)
        h = self.dropout(h) # Dropout

        # AdaGN projection of embedding -> chunkate into scale & shift
        c = self.embedding_mlp(embedding_vector) # B, C
        scale, shift = torch.chunk(c,chunks=2,dim=-1) # chunkate along channel dim
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]

        # Second path: norm (AdaGN) -> act -> conv
        h = self.norm2(h) * (1 + scale) + shift
        h = self.act2(h)
        h = self.conv2(h) # h has ch_out channels

        # Residual path: norm -> conv
        res = self.norm_res(x)
        res = self.res_proj(res)

        return h + res

class DownBlock(nn.Module): 
    """One block of the UNet encoder: Resblock x2 - (Att) - save skip - Downsample.
       returns x, skip."""
    def __init__(self, config, ch_in: int, ch_out: int, use_att: bool):
        super().__init__()

        self.resblock1 = ResBlock(config, ch_in, ch_out)
        self.resblock2 = ResBlock(config, ch_out, ch_out)
        self.att = AttentionBlock(config, ch_out) if use_att else nn.Identity()

        self.downsample = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=2, padding=1) # this halves the spatial dim

    def forward(self, x: torch.Tensor, embedding_vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x is (B,C,H,W)
        # embedding vector is (B,embvector_dim)

        x = self.resblock1(x, embedding_vector)
        x = self.resblock2(x, embedding_vector)
        x = self.att(x)

        skip = x # save skip before downsampling

        x = self.downsample(x)

        return x, skip

class UpBlock(nn.Module):
    """One block of the UNet decoder: Upsample - concat skip - Resblock x2 - (Att)"""
    def __init__(self, config, ch_in: int, ch_out: int, use_att: bool):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(ch_in, ch_in, kernel_size=4, stride=2, padding=1) # this doubles the spatial dim

        self.resblock1 = ResBlock(config, ch_in * 2, ch_out) # ch_in *2 because of the skip connection
        self.resblock2 = ResBlock(config, ch_out, ch_out)
        self.att = AttentionBlock(config, ch_out) if use_att else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor, embedding_vector: torch.Tensor) -> torch.Tensor:
        # x is (B,C,H,W)
        # skip.shape = x.shape after upsampling, so skip must have ch_in channels
        # embedding vector is (B,embvector_dim)

        x = self.upsample(x)

        torch._assert(skip.shape == x.shape, f"Skip shape missmatch! x: {x.shape}, skip: {skip.shape}")
        h = torch.cat([skip, x], dim=1) # concat across channel dim

        h = self.resblock1(h, embedding_vector)
        h = self.resblock2(h, embedding_vector)
        h = self.att(h)

        return h

class UNet(nn.Module):
    """Complete UNet, does not create embedding vector (c + t_emb) by itself."""
    def __init__(self, config):
        super().__init__()
        # att_depth = number of layers without att (from top down)

        def verb(text: str):
            """Prints if config.verbose == True"""
            if config.verbose:
                print(text)

        verb("\n=◈= UNET =◈=\n")
        verb("...CONFIG...")
        verb(f"ch_in: {config.ch_in}, ch: {config.ch}, ch_out: {config.ch_out}")
        verb(f"n_layers: {config.n_layers}, att_depth: {config.att_depth}, bottleneck_size: {config.bottleneck_size}, double_channels_in_bottleneck: {config.double_channels_in_bottleneck} ")
        verb(f"dropout: {config.dropout}")

        ## embedding MLP
        self.emb_mlp = nn.Sequential(
            nn.Linear(config.embvector_dim, config.embvector_dim),
            config.activation_function(),
            nn.Linear(config.embvector_dim, config.embvector_dim),
        )
        self.emb_proj = nn.Linear(config.embvector_dim, config.ch) # Early injection in UNet additional to AdaGN in ResBlocks
        
        ## Inconv
        self.inconv = nn.Conv2d(config.ch_in, config.ch, kernel_size=3, padding=1)
        verb(f"Inconv: ({config.ch_in} -> {config.ch})")

        ## ENCODER
        verb("\n...ENCODER... ")

        self.encoder = nn.ModuleList()
        ci = config.ch
        co = config.ch
        for i in range(config.n_layers):
            use_att = (i >= config.att_depth)
            self.encoder.append(DownBlock(config, ch_in=ci, ch_out=co, use_att=use_att))
            verb(f"DownBlock {i}: ch_in={ci} | ch_out={co} | Attention={use_att} | Spatial: {config.input_spatial * 0.5 ** (i)} -> {config.input_spatial * 0.5 ** (i+1)}      [Skip has {co} channels.]")
            ci = co
            if i != config.n_layers-1:
                co *= 2

        ## BOTTLENECK
        verb("\n...BOTTLENECK... ")
        
        d = 2 if config.double_channels_in_bottleneck else 1
        assert config.bottleneck_size > 0, "Bottleneck size cannot be 0."
        self.bottleneck = nn.ModuleList()
        for i in range(config.bottleneck_size):
            if i == 0:
                co2 = co * d
            elif i == config.bottleneck_size-1:
                co2 = co // d
            else:
                co2 = co

            self.bottleneck.append(ResBlock(config, ch_in=co, ch_out=co2, depthwise_separable=config.use_DSConvs))
            verb(f"**Bottleneck Resblock: {co}->{co2} {'(Depthwise Separable!)' if config.use_DSConvs else ''}")
            self.bottleneck.append(AttentionBlock(config, dim=co2))

            co = co2 # next block input = this block output


        verb(f"( Resblock + AttentionBlock ) x {config.bottleneck_size} | Spatial: {config.input_spatial * 0.5 ** config.n_layers} | Channel doubling inside bottleneck (hidden): {config.double_channels_in_bottleneck}")
        
        ## DECODER

        verb("\n...DECODER...")
        
        self.decoder = nn.ModuleList()
        ci = co
        co = co // 2
        for i in reversed(range(config.n_layers)):
            use_att = (i >= config.att_depth)

            if i == 0:
                co = config.ch
            else:
                co = ci //2

            self.decoder.append(UpBlock(config, ch_in=ci, ch_out=co, use_att=use_att))
            verb(f"UpBlock {i}: ch_in={ci} | ch_out={co} | Attention={use_att} | Spatial: {config.input_spatial * 0.5 ** (i+1)} -> {config.input_spatial * 0.5 ** (i)}      [Skip must have {ci} channels.]")
            ci = co

        
        # OutNorm - OutAct - OutConv
        self.outnorm = nn.GroupNorm(num_groups=min(32,config.ch), num_channels=config.ch)
        self.outact = config.activation_function()
        self.outconv = nn.Conv2d(config.ch, config.ch_out, kernel_size=3, padding=1)

        verb("\nOutNorm - OutAct - OutConv")

        
        if config.do_zero_init:
            # Zero-init the final conv to start with zero prediction
            nn.init.zeros_(self.outconv.weight)
            nn.init.zeros_(self.outconv.bias)

        if config.verbose:
            # Total parameters
            self.n_total_params = [sum(p.numel() for p in self.encoder.parameters()), sum(p.numel() for p in self.bottleneck.parameters()), sum(p.numel() for p in self.decoder.parameters()), sum(p.numel() for p in self.parameters())]
            # Trainable parameters only
            self.n_trainable_params = [sum(p.numel() for p in self.encoder.parameters() if p.requires_grad), sum(p.numel() for p in self.bottleneck.parameters() if p.requires_grad), sum(p.numel() for p in self.decoder.parameters() if p.requires_grad), sum(p.numel() for p in self.parameters() if p.requires_grad)]

            verb("\n...PARAMETERS...")
            verb(f"Encoder: [{round(self.n_total_params[0]/self.n_total_params[3]*100,2)}%] {self.n_total_params[0]:_} (Learnable: {self.n_trainable_params[0]:_})".replace("_", "'"))
            verb(f"Bottleneck: [{round(self.n_total_params[1]/self.n_total_params[3]*100,2)}%] {self.n_total_params[1]:_} (Learnable: {self.n_trainable_params[1]:_})".replace("_", "'"))
            verb(f"Decoder: [{round(self.n_total_params[2]/self.n_total_params[3]*100,2)}%] {self.n_total_params[2]:_} (Learnable: {self.n_trainable_params[2]:_})".replace("_", "'"))
            verb(f"Total: {self.n_total_params[3]:_} (Learnable: {self.n_trainable_params[3]:_})".replace("_", "'"))
            verb("\n=◈========◈=")

    def forward(
        self,
        x: torch.Tensor,
        embedding_vector: torch.Tensor,
        return_bottleneck: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        
        emb = self.emb_mlp(embedding_vector) # emb is B, embevector_dim

        x = self.inconv(x)
        x = x + self.emb_proj(emb)[:, :, None, None]

        ## Encoder
        skips = []
        for block in self.encoder:
            x, skip = block(x, emb)
            skips.append(skip)
        
        ## Bottleneck
        for block in self.bottleneck:
            if isinstance(block, ResBlock):
                x = block(x, emb)
            else:
                x = block(x) # Attention blocks don't have embeddingvector as input

        bottleneck_activations = x
        
        ## Decoder
        for block, skp in zip(self.decoder, reversed(skips)):
            x = block(x, skp, emb)
        
        x = self.outnorm(x)
        x = self.outact(x)
        x = self.outconv(x)

        return optional_bottleneck_output(x, bottleneck_activations, return_bottleneck)


# =============================================================================================
@torch.fx.wrap
def optional_bottleneck_output(output, bottleneck_activations, return_bottleneck):
    if return_bottleneck:
        return output, bottleneck_activations
    return output


@torch.fx.wrap # torch FX treats it as a black box, which avoids all tracing issues.
def sinusoidial_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    t: (B,) tensor
    dim: embedding dimension (must be even)
    returns: (B, dim)
    """

    assert dim % 2 == 0, f"dim must be divisible by two: {dim}"
    t = t.float()

    # convert t from [0,1] to [0,1000]
    t = t * 1000 

    half = dim // 2
    device = t.device
    dtype = t.dtype

    freqs = max_period ** (-2*torch.arange(half) / dim) # (half,)
    freqs = freqs = freqs.to(device=device, dtype=dtype)

    args = t[:, None] * freqs[None, :]  # (B,1) * (1,half) -> (B, half)

    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding # returns embedding vectors (B, dim)


class UNetPipeline(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embvector_dim = config.embvector_dim

        if config.time_emb_type == "table":
            self.t_emb_table = nn.Embedding(num_embeddings=config.T, embedding_dim=config.embvector_dim)
        else:
            self.t_emb_table = None
        
        self.null_class_id = config.n_classes
        self.class_emb_table = nn.Embedding(num_embeddings=config.n_classes+1, embedding_dim=config.embvector_dim)
        # n_classes += 1 embedding because the vector at index -1 of the embedding table is the null embedding.
        # -> UNet has its own emb_mlp
        self.net = UNet(config)
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        classids: torch.Tensor,
        return_bottleneck: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: # a classidx of -1 for the null embedding
        # x is (B,C,H,W)
        # classidx is (B,)

        classids = torch.where(classids<0, torch.full_like(classids, self.null_class_id), classids) # all negative ids become the null token

        t = t.float()
        if self.t_emb_table is not None:
            t_emb = self.t_emb_table(t)
        else:
            t_emb = sinusoidial_embedding(t.float(), self.embvector_dim)
        c_emb = self.class_emb_table(classids)
        embvector = c_emb + t_emb

        return self.net(x, embvector, return_bottleneck=return_bottleneck)

