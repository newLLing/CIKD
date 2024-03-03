from functools import partial
import torch
import torch.nn as nn
from vit import PatchEmbed, Block
from util.pos_embed import get_2d_sincos_pos_embed


class ALIViT(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, drop_path=0.1,
                 embed_dim=1024, depth=24, num_heads=16, last_heads=12,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()

        # --------------------------------------------------------------------------
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)
        self.last_heads = last_heads
        self.blocks = nn.ModuleList([
                                        Block(embed_dim, self.last_heads, mlp_ratio, qkv_bias=True, qk_scale=None,
                                              norm_layer=norm_layer)]+[
                                        Block(embed_dim, num_heads, mlp_ratio, drop_path=drop_path, qkv_bias=True,
                                              qk_scale=None, norm_layer=norm_layer)
                                        for i in range(depth - 2)] + [
                                        Block(embed_dim, self.last_heads, mlp_ratio, qkv_bias=True, qk_scale=None,
                                              norm_layer=norm_layer)])
        self.norm = norm_layer(embed_dim)

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        count = 0
        for blk in self.blocks:
            if count == 0:
                qk_1, vv_1 = blk(x, return_relation=True)
            count += 1
            if count == 12:
                qk, vv = blk(x, return_relation=True)
                return qk_1, vv_1, qk, vv
            else:
                x = blk(x)
        return x

    def forward_kd_loss(self, pred, teacher_out):
        loss = nn.KLDivLoss(reduction="none")(pred.log(), teacher_out).sum(-1)
        return loss.mean()
    def forward(self, imgs, teacher_out):
        qk_1, vv_1, qk, vv = self.forward_encoder(imgs)
        qk_1_loss = self.forward_kd_loss(qk_1, teacher_out[0])
        vv_1_loss = self.forward_kd_loss(vv_1, teacher_out[1])
        qk_11_loss = self.forward_kd_loss(qk, teacher_out[2])
        vv_11_loss = self.forward_kd_loss(vv, teacher_out[3])
        qk_loss = qk_11_loss + qk_1_loss
        vv_loss = vv_11_loss + vv_1_loss
        return qk_loss, vv_loss


def vit_tiny_patch16(**kwargs):
    model = ALIViT(
        patch_size=16, embed_dim=192, depth=12, num_heads=6, drop_path=0.1, last_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small_patch16(**kwargs):
    model = ALIViT(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, drop_path=0.1, last_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base_patch16(**kwargs):
    model = ALIViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, drop_path=0.1, last_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
