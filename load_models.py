from models import *
from gudhi.representations.vector_methods import Atol

def load_model(model, device, num_classes, grid_size, patch_size, depth=5, embed_dim=192):
    
    assert model in ['gin', 'gin_assisted_sum', 'gin_assisted_concat','xpert'], 'Model name not recognized'

    if model == 'xpert':
        model = Transformer(
                embed_dim=embed_dim,
                depth=depth,
                num_heads=8,
                grid_size=grid_size,
                patch_size=patch_size,
                mlp_ratio=4.0,
                qkv_bias=False,
                qk_scale=None,
                cls_token=True,
                token='conv',
                pos_type='sincos',
                drop_rate=0.5,
                attn_drop_rate=0.2,
                drop_path_rate=0.2,
                norm_layer=torch.nn.LayerNorm,
                init_std=0.02,
                classification_head=True,
                num_classes=num_classes
                ).to(device)
        
    elif model == 'gin':
        dim_h = 64
        model = GIN(dim_h, num_node_features=1, num_classes=num_classes).to(device)

    elif model == 'gin_assisted_concat':
        transformer = Transformer(
                        embed_dim=192,
                        depth=5,
                        num_heads=8,
                        grid_size=grid_size,
                        patch_size=patch_size,
                        mlp_ratio=4.0,
                        qkv_bias=False,
                        qk_scale=None,
                        cls_token=True,
                        token='conv',
                        pos_type='sincos',
                        drop_rate=0.5,
                        attn_drop_rate=0.2,
                        drop_path_rate=0.2,
                        norm_layer=torch.nn.LayerNorm,
                        init_std=0.02,
                        classification_head=False,
                        ).to(device)

        dim_h = 64
        model = GIN_assisted(transformer, dim_h, num_node_features=1, num_classes=num_classes, assist='concat').to(device)
    
    elif model == 'gin_assisted_sum':
        transformer = Transformer(
                        embed_dim=192,
                        depth=5,
                        num_heads=8,
                        grid_size=grid_size,
                        patch_size=patch_size,
                        mlp_ratio=4.0,
                        qkv_bias=False,
                        qk_scale=None,
                        cls_token=True,
                        token='conv',
                        pos_type='sincos',
                        drop_rate=0.5,
                        attn_drop_rate=0.2,
                        drop_path_rate=0.2,
                        norm_layer=torch.nn.LayerNorm,
                        init_std=0.02,
                        classification_head=False,
                        ).to(device)

        dim_h = 64
        model = GIN_assisted(transformer, dim_h, num_node_features=1, num_classes=num_classes, assist='sum').to(device)

    return model


def load_model_orbit(model, config):

    if model == 'xpert':
        model = Transformer_orbit(
                embed_dim=config['embed_dim'],
                depth=config['depth'],
                num_heads=config['num_heads'],
                grid_size=config['grid_size'],
                patch_size=config['patch_size'],
                mlp_ratio=4.0,
                qkv_bias=False,
                qk_scale=None,
                cls_token=True,
                pos_type='sincos',
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.0,
                norm_layer=nn.LayerNorm,
                init_std=0.02,
                n_classes=5
                )

    elif model == 'persformer':
        from gdeep.topology_layers import PersformerWrapper
        from gdeep.topology_layers.persformer_config import PoolerType

        model = PersformerWrapper(
            num_attention_layers=config['depth'],
            num_attention_heads=config['num_heads'],
            input_size= 2 + 2,
            output_size=5,
            pooler_type=PoolerType.ATTENTION,
            hidden_size=config['embed_dim'],
            intermediate_size=config['embed_dim'],
            use_skip_connections_for_persformer_blocks=True

        )

    return model
