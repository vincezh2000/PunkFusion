import os
from PIL import Image
from torch.utils.data import Dataset

class PunkImgDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, img_name)
                            for img_name in os.listdir(root_dir)
                            if img_name.endswith('.png')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

# Example usage:
# Assuming tf is your set of transformations, and PUNK_PATH is the directory path.
# dataset = PunkImgDataset(root_dir=PUNK_PATH, transform=tf)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=20)
