import numpy as np
import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    ToTensord,
    SpatialPadd,
    CastToTyped,
    DivisiblePadd
)
from monai.data import MetaTensor
from dafne_inference.transforms import PreprocessAnisotropy


def run_inference(model_obj, data_dict: dict) -> dict:
    input_image = data_dict['image']

    if not input_image.shape[0] < input_image.shape[1]:
        input_image = np.ascontiguousarray(np.moveaxis(input_image, -1, 0))

    try:
        affine_numpy = data_dict['affine']
    except KeyError:
        resolution = data_dict['resolution'] # this is always present
        if len(resolution) < 3:
            pixdim = (resolution[0], resolution[1], 1.0)
        else:
            pixdim = (resolution[0], resolution[1], resolution[2])
        affine_numpy = np.diag([*pixdim, 1.0]).astype(np.float64)
    
    data = { 'image': MetaTensor(input_image, affine = affine_numpy) }

    net_metadata = model_obj.metadata['net_metadata']
    dyn_model = net_metadata['use_dynamic']
    spacing = net_metadata['median_spacing']
    spatial_dims = net_metadata['spatial_dims']
    labels_name = net_metadata['labels_name']

    if not dyn_model:
        transf_list = [
            EnsureChannelFirstd(keys=['image'], channel_dim='no_channel'),
            PreprocessAnisotropy(keys=['image'], target_spacing=spacing,
                                 model_mode=None, spatial_dims=spatial_dims),
            DivisiblePadd(keys=['image'], k=32),
            ToTensord(keys=['image'])
        ]
    else:
        transf_list = [
            EnsureChannelFirstd(keys=['image'], channel_dim='no_channel'),
            PreprocessAnisotropy(keys=['image'], target_spacing=spacing,
                                 model_mode=None, spatial_dims=spatial_dims),
            SpatialPadd(keys=['image'],
                        spatial_size=net_metadata['patch_size'],
                        method="symmetric"),
            CastToTyped(keys=['image'], dtype=np.float32),
            ToTensord(keys=['image'])
        ]

    data_processed = Compose(transf_list)(data)
    img_tensor = data_processed['image']

    model_obj.model.eval()
    with torch.no_grad():
        if spatial_dims == 3:
            img_tensor = img_tensor.unsqueeze(0).to(model_obj.device)
            output = model_obj.model(img_tensor)
            pred_torch = torch.argmax(output, dim=1)
            pred_vol = pred_torch[0].detach().cpu().numpy().astype(np.int8)

        elif spatial_dims == 2:
            pred_vol = []
            depth = img_tensor.shape[1]
            for i in range(depth):
                slice_torch = img_tensor[:, i, :, :].unsqueeze(0).to(model_obj.device)
                output = model_obj.model(slice_torch)
                pred_torch = torch.argmax(output, dim=1)
                pred_vol.append(pred_torch[0].detach().cpu().numpy().astype(np.int8))
            pred_vol = np.stack(pred_vol, axis=0)

    return {label: (pred_vol == i + 1).astype(np.int8)
            for i, label in enumerate(labels_name)}
