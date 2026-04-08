import numpy as np
from dataclasses import asdict
from abc import ABC, abstractmethod

from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    ToTensord,
    CastToTyped,
    SpatialPadd,
    RandCropByPosNegLabeld,
    DivisiblePadd
    )
from dafne_inference.transforms import PreprocessAnisotropy, MapTransformLoadData


def build_transform_list(keys:list,
                         median_spacing:list, 
                         spatial_dims:int=2
                         ) -> Compose:
    
    if spatial_dims == 3: 
        pipeline = [
            MapTransformLoadData(keys=keys, spatial_dims=3),
            EnsureChannelFirstd(keys=['image', 'mask'], channel_dim='no_channel'),
            PreprocessAnisotropy(keys=['image', 'mask'], 
                                 target_spacing=median_spacing,
                                 model_mode='train'),
            ToTensord(keys=['image', 'mask'])
        ]
    
    else: 
        pipeline = [
            MapTransformLoadData(keys=keys, spatial_dims=2),
            EnsureChannelFirstd(keys=['image', 'mask'], channel_dim='no_channel'),
            PreprocessAnisotropy(keys=['image', 'mask'], 
                                 target_spacing=median_spacing,
                                 model_mode='train',
                                 spatial_dims=2),
            ToTensord(keys=['image', 'mask']), 
            DivisiblePadd(keys=['image', 'mask'], k=32)
        ]
    
    return Compose(pipeline)


def build_transforms_dynunet(keys: list,
                             patch_size: list, 
                             target_spacing: list,
                            )-> Compose:

    pipeline = [
        MapTransformLoadData(keys=keys, spatial_dims=3),
        EnsureChannelFirstd(keys=['image', 'mask'], channel_dim='no_channel'),
        PreprocessAnisotropy(keys=['image', 'mask'], 
                                                target_spacing=target_spacing,
                                                model_mode='train',
                                                spatial_dims=3),
        SpatialPadd(keys=keys, spatial_size=patch_size, method="symmetric"),
        RandCropByPosNegLabeld(
                keys=keys,
                label_key="mask",
                spatial_size=patch_size,
                pos=3, neg=1, num_samples=4, 
                image_key="image", image_threshold=0,
            ),
        CastToTyped(keys=keys, dtype=(np.float32, np.uint8)),
        ToTensord(keys=keys)
        ]
    return Compose(pipeline)