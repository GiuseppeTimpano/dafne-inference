import numpy as np
import torch 

from skimage.transform import resize
from monai.transforms.utils import generate_spatial_bounding_box

from monai.transforms import (MapTransform, 
                            CropForegroundd, 
                            NormalizeIntensity, 
                            SpatialCrop)
from .utils import resample_image


#function with image and mask as key
class MapTransformLoadData(MapTransform):
    '''
    Custom MONAI Transform to handle hybrid data loading strategies.
    It allow the pipeline to seamlessly handle both legacy .npz 
    archives (containing paired image/mask) and standard medical 
    formats (NIfTI, DICOM) stored in separate files.
    '''
    def __init__(self, keys, allow_missing_keys=False, spatial_dims:int=3):
        '''
        Args:
            keys (list): keys to processing in the data dictionary
            allow_missing_keys: it does not raise exception if key is missing
        '''
        super().__init__(keys, allow_missing_keys)
        self.spatial_dims = spatial_dims

    def __call__(self, data):
        '''
        Apply the transform to one sample (dictionary)
        '''
        d = dict(data)
        image = d['image']
        if image.ndim == 3:
            image = np.ascontiguousarray(np.moveaxis(image, -1, 0))
        masks = d['mask']
        mask = np.zeros_like(image, dtype=np.uint8)
        resolution = d['resolution']
        index = data.get('index', None)

        try:
            mask_keys = sorted(k for k in masks.keys())
            for i, k in enumerate(mask_keys):
                m = masks[k].astype(np.uint8)
                if m.ndim == 3:
                    m = np.ascontiguousarray(np.moveaxis(m, -1, 0))
                mask[m > 0] = i + 1

            if self.spatial_dims == 2 and image.ndim == 3:
                image = image[index]
                mask = mask[index]
                current_res = np.array([resolution[0], resolution[1]], dtype=np.float32)
            elif self.spatial_dims == 2:
                current_res = np.array([resolution[0], resolution[1]], dtype=np.float32)
            elif self.spatial_dims == 3:
                current_res = np.array([resolution[2], resolution[0], resolution[1]], dtype=np.float32)
            
            d['mask'] = mask
            d['image'] = image
            d['image_meta_dict'] = {
                "pixdim": np.array([1, *current_res], dtype=np.float32)
            }
            d['mask_meta_dict'] = {
                "pixdim": np.array([1, *current_res], dtype=np.float32)
            }
        except Exception as e: 
            print(f"Error during volume loading {e}")
        
        return d

class PreprocessAnisotropy(MapTransform):
    def __init__(self, 
                 keys, 
                 target_spacing,
                 clip_values=None, 
                 normalize_values=None, 
                 model_mode="train",
                 spatial_dims:int=3):
        super().__init__(keys)
        
        self.target_spacing = target_spacing
        self.spatial_dims = spatial_dims
        self.keys = keys

        self.low = clip_values[0] if clip_values else 0
        self.high = clip_values[1] if clip_values else 0
        self.mean = normalize_values[0] if normalize_values else 0
        self.std = normalize_values[1] if normalize_values else 1

        self.training = (model_mode == 'train')

        self.crop_foreg = CropForegroundd(keys=['image', 'mask'], source_key='image', allow_smaller=True, allow_missing_keys=True)
        self.normalize_intensity = NormalizeIntensity(nonzero=True, channel_wise=True)

    def calculate_new_shape(self, spacing, shape):
        aspect_ratio = np.array(spacing) / self.target_spacing
        new_shape = np.round(aspect_ratio * np.array(shape)).astype(int)
        return new_shape
    
    def check_anisotrophy(self, spacing):
        def check(s):
            if s is None:
                return False
            return np.max(s) / np.min(s) >= 3
        return check(spacing) or check(self.target_spacing)
    
    def __call__(self, data):
        d = dict(data)
        image = d['image']

        has_mask = 'mask' in self.keys and 'mask' in d
        label = d['mask'] if has_mask else None

        current_spacing = d['image_meta_dict']['pixdim']
        current_spacing = current_spacing[1:] # no channel spacing
        image_spacing = np.array(current_spacing).tolist()

        if self.training:
            cropped_data = self.crop_foreg({"image": image, "mask": label} if has_mask else {"image": image})
            if 0 in cropped_data["image"].shape[1:]:
                pass
            else:
                image = cropped_data["image"]
                if has_mask:
                    label = cropped_data["mask"]
        else:
            d["original_shape"] = np.array(image.shape[1:])
            box_start, box_end = generate_spatial_bounding_box(image, allow_smaller=True)
            temp_image = SpatialCrop(roi_start=box_start, roi_end=box_end)(image)
            if 0 in temp_image.shape[1:]:
                d["bbox"] = np.vstack([box_start, box_end])
                d["crop_shape"] = np.array(image.shape[1:])
            else:
                image = temp_image
                if has_mask:
                    label = SpatialCrop(roi_start=box_start, roi_end=box_end)(label)
                d["bbox"] = np.vstack([box_start, box_end])
                d["crop_shape"] = np.array(image.shape[1:])

        original_shape = image.shape[1:]

        resample_flag = False
        anisotrophy_flag = False

        if isinstance(image, torch.Tensor): image = image.numpy()
        if has_mask and isinstance(label, torch.Tensor): label = label.numpy()

        if 0 not in original_shape and not np.allclose(self.target_spacing, image_spacing, atol=1e-4):
            resample_flag = True
            anisotrophy_flag = self.check_anisotrophy(image_spacing)
            resample_shape = self.calculate_new_shape(image_spacing, original_shape)

            if self.spatial_dims == 3:
                image = resample_image(image, resample_shape, anisotrophy_flag)
                if has_mask:
                    from .utils import _resample_prediction
                    label = np.expand_dims(
                        _resample_prediction(label[0], tuple(resample_shape), anisotrophy_flag), 0
                    ).astype(np.uint8)
            elif self.spatial_dims == 2:
                resized_channels = []
                for img_c in image:
                    res = resize(img_c, resample_shape, order=3, mode="edge",
                                 cval=0, clip=True, anti_aliasing=False)
                    resized_channels.append(res)
                image = np.stack(resized_channels, axis=0)
                if has_mask:
                    resized_mask_channels = []
                    for mask_c in label:
                        mask_res = resize(mask_c, resample_shape, order=0, mode="edge",
                                          cval=0, clip=True, anti_aliasing=False)
                        resized_mask_channels.append(mask_res.astype(np.uint8))
                    label = np.stack(resized_mask_channels, axis=0)

        d["resample_flag"] = resample_flag
        d["anisotrophy_flag"] = anisotrophy_flag
        d["resample_shape"] = np.array(resample_shape if resample_flag else original_shape)

        if self.low != 0 or self.high != 0:
            image = np.clip(image, self.low, self.high)
            image = (image - self.mean) / self.std
        else:
            image = self.normalize_intensity(image.copy())

        d["image"] = image
        if has_mask:
            d["mask"] = label

        return d