# ViL for Echo 2D Segmentation

This is a PyTorch implementation of ViL for Echo 2D Segmentation.

## Requirements

This code is tested with PyTorch 1.7.0 and Python 3.7.
The code is based on [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning).
The code is based on [ViT-pytorch](https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer).

To install the requirements, run:
```bash
pip install -r requirements.txt
```
## Dataset
The dataset is available at [Google Drive](https://drive.google.com/drive/folders/1-1ZYzkzKJx0-KYzkzKJx0-KYzkzKJx0-KYzkzKJx0-KYzkzKJx0-KYzkzKJx0-KYzkzKJx0-KYzkzKJx0-KYzkzKJx0-KYz
)

## Training
To train the model, run:
```bash
python train.py
```
The training script takes the following arguments:
```bash
--data_dir: path to the dataset
--batch_size: batch size
--num_workers: number of workers
--lr: learning rate
--num_epochs: number of epochs
--checkpoint_dir: path to the checkpoint directory
--checkpoint_path: path to the checkpoint file
--resume_from_checkpoint: resume from the checkpoint file
--resume_from_checkpoint_path: resume from the checkpoint file
```
The training script will save the checkpoint file to the checkpoint directory.

## Testing
To test the model, run:
```bash
python test.py
```
