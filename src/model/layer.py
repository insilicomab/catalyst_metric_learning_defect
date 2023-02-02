import torch
from catalyst.contrib.layers import (
    AdaCos,
    AMSoftmax,
    ArcFace,
    ArcMarginProduct,
    CosFace,
    CurricularFace,
    SubCenterArcFace,
)


def get_layer(
    layer_name: str,
    embedding_size: int,
    num_classes: int,
    ) -> torch.nn.Module:
    if layer_name == 'AdaCos':
        return AdaCos(
            in_features=embedding_size, 
            out_features=num_classes
        )
    elif layer_name == 'AMSoftmax':
        return AMSoftmax(
            in_features=embedding_size, 
            out_features=num_classes
        )
    elif layer_name == 'ArcFace':
        return ArcFace(
            in_features=embedding_size, 
            out_features=num_classes
        )
    elif layer_name == 'ArcMarginProduct':
        return ArcMarginProduct(
            in_features=embedding_size, 
            out_features=num_classes
        )
    elif layer_name == 'CosFace':
        return CosFace(
            in_features=embedding_size, 
            out_features=num_classes
        )
    elif layer_name == 'CurricularFace':
        return CurricularFace(
            in_features=embedding_size, 
            out_features=num_classes
        )
    elif layer_name == 'SubCenterArcFace':
        return SubCenterArcFace(
            in_features=embedding_size, 
            out_features=num_classes
        )
    else:
        raise ValueError(f'Unknown optimizer: {layer_name}')