import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# 1. 时间步编码 (用于告知模型当前处于扩散的哪一步)
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# 2. 基础卷积块
class Conv1dBlock(nn.Module):
    def __init__(self, inp_ch, out_ch, kernel_size, n_groups=8):
        super().__init__()

        # --- 核心修复逻辑 ---
        # 检查 out_ch 是否能被 n_groups 整除
        # 如果不能，则自动寻找一个合适的组数（最简单是设为 1，或者判断 14 能被 2 或 7 整除）
        if out_ch % n_groups != 0:
            if out_ch % 2 == 0:
                actual_groups = 2  # 比如 14 可以被 2 整除
            else:
                actual_groups = 1  # 实在不行就设为 1
        else:
            actual_groups = n_groups
        # -------------------

        self.block = nn.Sequential(
            nn.Conv1d(inp_ch, out_ch, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(actual_groups, out_ch),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


# 3. 核心 U-Net 架构
class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        action_dim=14,
        global_cond_dim=9,
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
    ):
        super().__init__()

        # 1. 时间步编码
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )

        # 2. 下采样维度逻辑: [14, 256, 512, 1024]
        all_dims = [action_dim] + list(down_dims)
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        cond_dim = global_cond_dim + diffusion_step_embed_dim

        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        Conv1dBlock(dim_in, dim_out, kernel_size=5),
                        Conv1dBlock(dim_out, dim_out, kernel_size=5),
                        nn.Linear(cond_dim, dim_out),
                        (
                            nn.Conv1d(dim_out, dim_out, 3, stride=2, padding=1)
                            if not is_last
                            else nn.Identity()
                        ),
                    ]
                )
            )

        self.mid_block = nn.ModuleList(
            [
                Conv1dBlock(down_dims[-1], down_dims[-1], kernel_size=5),
                Conv1dBlock(down_dims[-1], down_dims[-1], kernel_size=5),
            ]
        )

        # 3. 上采样维度逻辑 (关键修正点)
        # 我们让上采样停留在 down_dims[0] (即 256)，而不是直接跳到 14
        self.up_modules = nn.ModuleList([])
        # 构造上采样路径: [(1024, 512), (512, 256), (256, 256)]
        up_in_out = []
        for i in range(len(down_dims) - 1, 0, -1):
            up_in_out.append((down_dims[i], down_dims[i - 1]))
        up_in_out.append((down_dims[0], down_dims[0]))  # 最后一层保持在 256 维

        for ind, (dim_in_up, dim_out_up) in enumerate(up_in_out):
            is_last = ind >= (len(up_in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        Conv1dBlock(
                            dim_in_up * 2, dim_out_up, kernel_size=5
                        ),  # *2 是因为 Skip Connection
                        Conv1dBlock(dim_out_up, dim_out_up, kernel_size=5),
                        nn.Linear(cond_dim, dim_out_up),
                        (
                            nn.ConvTranspose1d(
                                dim_out_up, dim_out_up, 4, stride=2, padding=1
                            )
                            if not is_last
                            else nn.Identity()
                        ),
                    ]
                )
            )

        # 4. 最终输出层: 从 256 映射到 14 (保持不变)
        self.final_conv = nn.Sequential(
            Conv1dBlock(down_dims[0], down_dims[0], kernel_size=5),
            nn.Conv1d(down_dims[0], action_dim, 1),
        )

    def forward(self, sample, timestep, global_cond):
        # sample 形状: (B, T, 14) -> 转换为 (B, 14, T) 以适配 Conv1d
        x = sample.moveaxis(-1, -2)

        timesteps_embed = self.diffusion_step_encoder(timestep)
        cond = torch.cat([global_cond, timesteps_embed], dim=-1)

        h = []
        for resnet, resnet2, cond_layer, downsample in self.down_modules:
            x = resnet(x)
            x = x + cond_layer(cond).unsqueeze(-1)
            x = resnet2(x)
            h.append(x)
            x = downsample(x)

        for m in self.mid_block:
            x = m(x)

        for resnet, resnet2, cond_layer, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)  # 跳跃连接拼接
            x = resnet(x)
            x = x + cond_layer(cond).unsqueeze(-1)
            x = resnet2(x)
            x = upsample(x)

        x = self.final_conv(x)
        return x.moveaxis(-1, -2)  # 返回 (B, T, 14)
