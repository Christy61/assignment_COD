import os
from PIL import Image
from torch.utils.data import Dataset

class Custom(Dataset):

    def __init__(self, 
                 base_dir,
                 split='train',
                 transform=None):
        super().__init__()
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, split, 'image')
        self._cat_dir = os.path.join(self._base_dir, split, 'GT')
        self.transform = transform

        self.filenames = [i_id.strip().split(".")[0] for i_id in os.listdir(self._image_dir)]
        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.filenames)))

    def __getitem__(self, index):
        img, label = self._make_img_gt_point_pair(index)
        if self.transform is not None:
            img, label = self.transform(img, label)
        return img, label

    def __len__(self):
        return len(self.filenames)
    
    def get_name(self):
        return self.filenames

    def _make_img_gt_point_pair(self, index):

        filename = self.filenames[index]

        _img = Image.open(self._image_dir + "/" + str(filename) + '.jpg').convert('RGB')
        _target = Image.open(self._cat_dir + "/" + str(filename) + '.png').convert("1")
        return _img, _target
    
