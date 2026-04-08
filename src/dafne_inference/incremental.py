import numpy as np
import torch

from monai.data import Dataset, DataLoader
from monai.losses import DiceCELoss
from monai.data.utils import pad_list_data_collate
from dafne_inference.transforms_builder import build_transform_list, build_transforms_dynunet


def compute_ewc_loss(model, fisher, params_snapshot, lambda_reg):
    ewc_loss = 0.0
    for name, param in model.named_parameters():
        if name in fisher and name in params_snapshot:
             ewc_loss += (fisher[name] * (param - params_snapshot[name]) ** 2).sum()
    return lambda_reg * ewc_loss


def run_incremental_learning(model_obj, trainingData:dict, trainingOutputs:dict, bs, minTrainImages):
    image_list = trainingData.get('image_list', [])
    if len(image_list) < minTrainImages:
        return
    
    #read model metadata
    net_metadata = model_obj.metadata['net_metadata']
    spatial_dims = net_metadata['spatial_dims']
    spacing = net_metadata['median_spacing']
    patch_size = net_metadata['patch_size']
    use_dynamic = net_metadata['use_dynamic']
    ewc_data = model_obj.metadata.get('ewc_data', None)
    device = model_obj.device
    model = model_obj.model
    resolution = trainingData.get('resolution', np.ones(spatial_dims))

    if spatial_dims == 3: #take mask list
        mask_list = [trainingOutputs[k] for k in sorted(trainingOutputs.keys())]
    else: 
        mask_list = list(trainingOutputs)
    
    assert len(image_list) == len(mask_list)
    samples = []
    for img, mask_dict in zip(image_list, mask_list):
         samples.append({
              'image': img,
              'mask': mask_dict,
              'resolution': resolution
         })
    
    if not use_dynamic: 
         transforms = build_transform_list(keys=['image', 'mask'], 
                                           median_spacing=spacing,
                                           spatial_dims=spatial_dims)
    else: 
         transforms = build_transforms_dynunet(keys=['image', 'mask'], 
                                               patch_size=patch_size,
                                               target_spacing=spacing)
    dataset = Dataset(data=samples, transform=transforms)
    dataloader = DataLoader(dataset=dataset, batch_size=bs, num_workers=0, collate_fn=pad_list_data_collate)

    scaler = torch.amp.GradScaler(enabled=True)
    optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=1e-3, 
                weight_decay=1e-4
            )
    
    criterion = DiceCELoss(to_onehot_y=True, softmax=True)
    n_epochs = 5
    
    fisher_t, snapshot_t = None, None
    lambda_reg = 0.5
    if ewc_data is not None:                                                                                                                                                   
        fisher_t   = {k: torch.tensor(v, dtype=torch.float32, device=device)                                                                                                   
                        for k, v in ewc_data['fisher'].items()}
        snapshot_t = {k: torch.tensor(v, dtype=torch.float32, device=device)                                                                                                   
                        for k, v in ewc_data['params_snapshot'].items()}
    
    for epoch in range(n_epochs):
         epoch_loss = 0.0 
         n_steps = 0
         for batch in dataloader:
              train_results = train_increment_one_epoch(
                   model, batch, device, scaler, optimizer,
                   criterion, fisher_t, snapshot_t, lambda_reg
              )
              epoch_loss += train_results['loss']
              n_steps += 1
         print(f"[IL] Epoch {epoch + 1}/{n_epochs}  loss: {epoch_loss / max(n_steps, 1):.4f}")

def train_increment_one_epoch(model, 
                    batch, 
                    device, 
                    scaler, 
                    optimizer, 
                    criterion,
                    fisher_t,
                    snapshot_t, 
                    lambda_reg): #typical pytorch training loop for one epoch incremental learning
    model.train()
    inputs = batch['image'].to(device)
    targets = batch['mask'].long().to(device)
    optimizer.zero_grad()
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if fisher_t is not None:
                loss = loss + compute_ewc_loss(model, fisher_t, snapshot_t, lambda_reg)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
    scaler.step(optimizer)
    scaler.update()
    return {'loss': loss.item()}