import os

import torch
import torchvision
import PIL.Image

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode, transform = None):
        self.root = root
        self.mode = mode
        self.transform = transform
        self.ys, self.im_paths, self.I = [], [], []

    def nb_classes(self):
        assert set(self.ys) == set(self.classes)
        return len(self.classes)

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, index):
        path = self.im_paths[index]
        try:
            with PIL.Image.open(path) as source:
                image = source.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        except Exception as exc:
            raise RuntimeError(f"Không đọc/xử lý được ảnh: {path}") from exc
        return image, self.ys[index]

    def get_label(self, index):
        return self.ys[index]

    def set_subset(self, I):
        self.ys = [self.ys[i] for i in I]
        self.I = [self.I[i] for i in I]
        self.im_paths = [self.im_paths[i] for i in I]


class CUBirds(BaseDataset):
    def __init__(self, root, mode, transform = None):
        self.root = os.path.join(root, "CUB_200_2011")
        self.mode = mode
        self.transform = transform
        if self.mode == 'train':
            self.classes = range(0,100)
        elif self.mode == 'eval':
            self.classes = range(100,200)
        else:
            raise ValueError(f"Split không hợp lệ: {mode!r}; cần train hoặc eval")
        
        BaseDataset.__init__(self, self.root, self.mode, self.transform)
        index = 0
        for i in torchvision.datasets.ImageFolder(root = 
                os.path.join(self.root, 'images')).imgs:
            # i[1]: label, i[0]: the full path to an image
            y = i[1]
            # fn needed for removing non-images starting with `._`
            fn = os.path.split(i[0])[1]
            if y in self.classes and fn[:2] != '._':
                self.ys += [y]
                self.I += [index]
                self.im_paths.append(i[0])
                index += 1

        if not self.im_paths:
            raise RuntimeError(f"Không tìm thấy ảnh cho split {mode!r} trong {self.root}")


def collate_pil_batch(batch):
    """Keep PIL images unstacked so the checkpoint processor can batch them."""
    images, labels = zip(*batch, strict=True)
    return list(images), torch.tensor(labels, dtype=torch.long)
