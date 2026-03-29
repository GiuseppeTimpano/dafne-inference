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
from monai.inferers import sliding_window_inference
from dafne_inference.transforms import PreprocessAnisotropy
from .utils import _center_crop, _resample_prediction


def run_inference(model_obj, data_dict: dict) -> dict:
    input_image = data_dict['image']

    # dafne sends (X, Y, Z) — convert to (Z, X, Y); Z (slices) is always the smallest axis
    if input_image.shape[2] < input_image.shape[0]:
        input_image = np.ascontiguousarray(np.moveaxis(input_image, -1, 0))

    try:
        affine_numpy = data_dict['affine']
        # affine column norms → spacing in (X, Y, Z) order (dafne convention)
        # reorder to (Z, X, Y) to match image axes after moveaxis(-1, 0)
        col_norms = np.sqrt((affine_numpy[:3, :3] ** 2).sum(axis=0))
        pixdim = np.array([col_norms[2], col_norms[0], col_norms[1]])
    except KeyError:
        resolution = data_dict['resolution']
        if len(resolution) < 3:
            pixdim = np.array([resolution[0], resolution[1]], dtype=np.float64)
        else:
            pixdim = np.array([resolution[2], resolution[0], resolution[1]], dtype=np.float64)
        affine_numpy = np.diag([*pixdim, 1.0]).astype(np.float64)

    data = {
        'image': MetaTensor(input_image, affine=affine_numpy),
        'image_meta_dict': {'pixdim': np.array([1.0, *pixdim], dtype=np.float32)}
    }

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
            # use sliding window for inference during inference
            output = sliding_window_inference(
                inputs=img_tensor,
                roi_size=net_metadata['patch_size'],
                sw_batch_size=4,
                overlap=0.25,
                predictor=model_obj.model
            )
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

    # Post-processing: return image to (X, Y, Z) space

    # Remove simmetric padding adding by DivisiblePadd/SpatialPadd
    resample_shape = tuple(data_processed['resample_shape'])
    pred_vol = _center_crop(pred_vol, resample_shape)

    # Inverse resample → from model spacing to original crop_shape
    if data_processed['resample_flag']:
        crop_shape = tuple(data_processed['crop_shape'])
        anisotrophy_flag = bool(data_processed['anisotrophy_flag'])
        pred_vol = _resample_prediction(pred_vol, crop_shape, anisotrophy_flag)

    # Insert to original volume using foreground's bbox 
    original_shape = tuple(data_processed['original_shape'])
    if pred_vol.shape != original_shape:
        bbox = data_processed['bbox']  # shape (2, ndim): [box_start, box_end]
        box_start, box_end = bbox[0].astype(int), bbox[1].astype(int)
        full_pred = np.zeros(original_shape, dtype=np.int8)
        full_pred[box_start[0]:box_end[0],
                  box_start[1]:box_end[1],
                  box_start[2]:box_end[2]] = pred_vol
        pred_vol = full_pred

    # Convert axes (Z, X, Y) → (X, Y, Z)
    pred_vol = np.moveaxis(pred_vol, 0, -1)

    return {label: (pred_vol == i + 1).astype(np.int8)
            for i, label in enumerate(labels_name)}
