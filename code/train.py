

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from engine import *
import cv2
import os
# from configs.config_setting import setting_config
from loss import Mask_DC_and_BCE_loss
import warnings
import json
import random
import albumentations as A
# 加载模型

from models.unet import UNet
from models.vision_lstm2 import VisionLSTM2
warnings.filterwarnings("ignore")


def correct_dims(*images):
    corr_images = []
    # print(images)
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)
    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images


def main():

    print('#----------GPU init----------#')
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    data_path = "/data/"

    train_loader = DataLoader(dataset=train_dataset, num_workers=8,
                              batch_size=2, pin_memory=True)


    test_loader = DataLoader(dataset=test_dataset, num_workers=8,
                             batch_size=1, pin_memory=False)

    print('#----------Prepareing Model----------#')

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    torch.distributed.init_process_group(backend="nccl")
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    seed_value = 42  # the number of seed
    np.random.seed(seed_value)  # set random seed for numpy
    random.seed(seed_value)  # set random seed for python
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
    torch.manual_seed(seed_value)  # set random seed for CPU
    torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
    torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
    torch.backends.cudnn.deterministic = True  # set random seed for convolution
    model = model.to(device)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))

    print('#----------Prepareing loss, opt, sch and amp----------#')
    pos_weight = torch.ones([1]).cuda(device=device) * 2
    criterion = Mask_DC_and_BCE_loss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, betas=(0.9, 0.999),
                                  weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=2,
        gamma=0.5,
        last_epoch=-1
    )
    step = 0


    print('#----------Training----------#')
    # 定义模型保存路径
    save_dir = f"./models/3/"
    os.makedirs(save_dir, exist_ok=True)
    for epoch in range(0, 30):
        bn = int(len(train_dataset))
        loss = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            step,
            device,
            bn
        )
        print("batch_num:", epoch)
        print("train_loss:{:.4f}".format(loss))
        # 保存模型
        path = os.path.join(save_dir, f"model_epoch_{epoch}.pth")
        torch.save(model.state_dict(), path, _use_new_zipfile_serialization=False)
        path = ".path" + str(epoch) + ".pth"
        torch.save(model.state_dict(), path, _use_new_zipfile_serialization=False)
    

if __name__ == '__main__':
    main()
