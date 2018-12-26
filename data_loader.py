from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from os import listdir, path
from PIL import Image


class CelebADataset(Dataset):
    def __init__(self, data_path, transform=lambda x:x):
        self.data_path = data_path
        self.transform = transform
        self.images = listdir(data_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img = Image.open(path.join(self.data_path, self.images[item]))
        return self.transform(img)

class CelebAWithIndexedTransformDataset(CelebADataset):
    def __init__(self, data_path, transform_function=None):
        """

        Args:
            data_path: path to the folder with the dataset
            transform_function: function that generates the transformation
            when given the index number
        """
        super(CelebAWithIndexedTransformDataset, self).__init__(data_path,
                                                                None)
        self._set_transform_by_index = transform_function
        def not_implemented(*args):
            raise NotImplementedError("The transformation function has "
                "not been yet defined. Please call first set_transform_by_index(index)")
        self.transform = not_implemented

    def set_transform_by_index(self, index):
        self.transform = self._set_transform_by_index(index)


def get_loader(data_path, crop_size, batch=16):
    def define_transformation_by_index(index):
        return T.Compose([
            T.CenterCrop(crop_size),
            T.Resize(2**(index + 2)),
            T.ToTensor(),
            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    dataset = CelebAWithIndexedTransformDataset(data_path,
                                                transform_function=define_transformation_by_index)
    data_loader = DataLoader(dataset, batch_size=batch, shuffle=True)
    return data_loader