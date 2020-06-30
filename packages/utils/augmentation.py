from albumentations import Compose, Resize, Normalize, Flip
from albumentations.pytorch import ToTensorV2


def get_transforms(cfg):
    train_transforms_list = [Resize(*cfg.RESIZE)]
    test_transforms_list = [Resize(*cfg.RESIZE)]

    if cfg.FLIP:
        train_transforms_list.append(Flip(p=cfg.FLIP_PROB))

    if cfg.NORMALIZE:
        train_transforms_list.append(Normalize())
        test_transforms_list.append(Normalize())

    train_transforms_list.append(ToTensorV2())
    test_transforms_list.append(ToTensorV2())
    

    train_transforms = Compose(train_transforms_list)
    test_transforms = Compose(test_transforms_list)
    return train_transforms, test_transforms
