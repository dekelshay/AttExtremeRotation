import torch
import numpy as np
import pandas as pd
from torch.utils import data
from torchvision.datasets.vision import VisionDataset
from PIL import Image
import os
import os.path
from torchvision import transforms
from PIL import ImageFile
from tqdm import tqdm


ImageFile.LOAD_TRUNCATED_IMAGES = True

class RotationDataset(VisionDataset):
    def __init__(self, root, loader, extensions=None, height=None, pairs_file=None, transform=None,
                 target_transform=None, Train=True):
        super(RotationDataset, self).__init__(root, transform=transform,
                                           target_transform=target_transform)
        # self.pairs = np.load(pairs_file, allow_pickle=True).item()  ## TO DO EXCHNAGED THIS
        self.pairs = load_pairs_from_csv( pairs_file)
        self.loader = loader
        self.extensions = extensions
        self.train = Train
        self.height = height

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            disctionary: img1, img2, rotation_x1, rotation_x2, rotation_y1, rotation_y2, path, path2
        """
        img1 = self.pairs[index]['img1']
        img2 = self.pairs[index]['img2']
        path = os.path.join(self.root, img1['path'])
        rotation_x1, rotation_y1 = img1['x'], img1['y']
        image1 = self.loader(path)
        if self.target_transform is not None:
            image1 = self.target_transform(image1)
        path2 = os.path.join(self.root, img2['path'])
        rotation_x2, rotation_y2 = img2['x'], img2['y']
        image2 = self.loader(path2)
        if self.target_transform is not None:
            image2 = self.target_transform(image2)

        return {
            'img1': image1,
            'rotation_x1': rotation_x1,
            'rotation_y1': rotation_y1,
            'img2': image2,
            'rotation_x2': rotation_x2,
            'rotation_y2': rotation_y2,
            'path': path,
            'path2': path2,
        }

    def __len__(self):
        if len(self.pairs) > 1000 and not self.train:
            return 1000
        return len(self.pairs)


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def load_pairs_from_csv(file_path):
    pair_df = pd.read_csv(file_path)
    pair_dict = []

    for count, item in tqdm(enumerate(pair_df.iterrows())):
        pair_dict.append( {'img1':{'path': item[1]['filenames0'] , 'x': item[1]['x0'], 'y': item[1]['y0']},
                           'img2':{'path': item[1]['filenames1'], 'x':  item[1]['x1'], 'y': item[1]['y1']},'translation':0.0  })



    return( pair_dict )


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def get_datasets(cfg):
    tr_dataset = RotationDataset(cfg.train.path, default_loader, '.png', height=cfg.height,
                              pairs_file=cfg.train.pairs_file,
                              transform=transforms.Compose(
                                  [transforms.Resize((int(cfg.height), int(cfg.height))), transforms.ToTensor(),
                                   transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                   ]),
                              target_transform=transforms.Compose(
                                  [transforms.Resize((int(cfg.height), int(cfg.height))), transforms.ToTensor(),
                                   transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                   ])
                              )
    te_dataset = RotationDataset(cfg.val.path, default_loader, '.png', height=cfg.height, pairs_file=cfg.val.pairs_file,
                              transform=transforms.Compose(
                                  [transforms.Resize((int(cfg.height), int(cfg.height))), transforms.ToTensor(),
                                   transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                   ]),
                              target_transform=transforms.Compose(
                                  [transforms.Resize((int(cfg.height), int(cfg.height))), transforms.ToTensor(),
                                   transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                   ]),
                              Train=False)
    return tr_dataset, te_dataset


def init_np_seed(worker_id):
    seed = torch.initial_seed()
    np.random.seed(seed % 4294967296)


def get_data_loaders(cfg):
    tr_dataset, te_dataset = get_datasets(cfg)
    train_loader = data.DataLoader(
        dataset=tr_dataset, batch_size=cfg.train.batch_size,
        shuffle=True, num_workers=cfg.num_workers, drop_last=True,
        worker_init_fn=init_np_seed)
    test_loader = data.DataLoader(
        dataset=te_dataset, batch_size=cfg.val.batch_size,
        shuffle=False, num_workers=cfg.num_workers, drop_last=False,
        worker_init_fn=init_np_seed)

    loaders = {
        "test_loader": test_loader,
        'train_loader': train_loader,
    }
    return loaders


if __name__ == "__main__":
    pass
