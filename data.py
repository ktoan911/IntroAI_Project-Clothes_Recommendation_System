import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ClotheDataset(Dataset):
    def __init__(self, datasets_path: str, transform=None, part="train"):
        datasets_dir = os.path.join(datasets_path, str(part))

        self.categories = [
            folder
            for folder in os.listdir(datasets_dir)
            if os.path.isdir(os.path.join(datasets_dir, folder))
        ]
        self.id2label = {}
        self.label2id = {}
        for i, cat in enumerate(self.categories):
            self.id2label[i] = cat
            self.label2id[cat] = i

        self.images = []
        self.labels = []

        for category in os.listdir(datasets_dir):
            category_dir = os.path.join(datasets_dir, category)

            try:
                if os.path.isdir(category_dir):
                    for img_file in os.listdir(category_dir):
                        if img_file.lower().endswith(".png"):
                            img_path = os.path.join(category_dir, img_file)
                            self.images.append(Image.open(img_path).copy())
                            self.labels.append(self.label2id[category])
            except Exception:
                continue
        self.transform = transform
        if not self.transform:
            self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
