from functools import partial
import timm.models.vision_transformer
import torch
import torch.nn as nn
from util.pos_embed import get_2d_sincos_pos_embed
from vit import PatchEmbed, Block
from models_ALI import ALIViT

class ViT(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=100,
                 drop_path=0.1,
                 embed_dim=192, depth=12, num_heads=12, last_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, **kwargs):
        super(ALIViT, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.last_heads = last_heads
        self.blocks = nn.ModuleList([
                                        Block(embed_dim, self.last_heads, mlp_ratio, qkv_bias=True, qk_scale=None,
                                              norm_layer=norm_layer)] + [
                                        Block(embed_dim, num_heads, mlp_ratio, drop_path=drop_path, qkv_bias=True,
                                              qk_scale=None, norm_layer=norm_layer)
                                        for i in range(depth - 2)] + [
                                        Block(embed_dim, self.last_heads, mlp_ratio, qkv_bias=True, qk_scale=None,
                                              norm_layer=norm_layer)])
        self.norm = norm_layer(embed_dim)
        self.fc_norm = nn.Identity()
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

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
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        outcome = x[:, 0]

        return outcome

    def forward_head(self, x, pre_logits: bool = False):
        x = self.fc_norm(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def vit_tiny_patch16(**kwargs):
    model = ViT(
        patch_size=16, embed_dim=192, depth=12, num_heads=6, drop_path=0.1, last_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small_patch16(**kwargs):
    model = ViT(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, drop_path=0.1, last_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base_patch16(**kwargs):
    model = ViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, drop_path=0.1, last_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
