import os
import torch
import torch.utils.data as data


class TIMIT(data.Dataset):
    training_file = 'training.pt'
    test_file = 'test.pt'
    val_file = 'val.pt'

    def __init__(self, root, mode="train"):
        self.root = os.path.expanduser(root)
        if mode != "train" and mode != "test" and mode != "val":
            raise RuntimeError("Wrong mode. Possible modes: train, test, val.")
        self.mode = mode  # training set, validation set or test set

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        if self.mode == "train":
            self.train_data, self.train_labels, self.train_lens = torch.load(
                os.path.join(self.root, self.training_file))
        elif self.mode == "test":
            self.test_data, self.test_labels, self.test_lens = torch.load(
                os.path.join(self.root, self.test_file))
        elif self.mode == "val":
            self.val_data, self.val_labels, self.val_lens = torch.load(
                os.path.join(self.root, self.val_file))

    def __getitem__(self, index):
        if self.mode == "train":
            points, target, lens = self.train_data[index], self.train_labels[index], self.train_lens[index]
        elif self.mode == "test":
            points, target, lens = self.test_data[index], self.test_labels[index], self.test_lens[index]
        elif self.mode == "val":
            points, target, lens = self.val_data[index], self.val_labels[index], self.val_lens[index]

        return points, target, lens

    def __len__(self):
        if self.mode == "train":
            return len(self.train_data)
        elif self.mode == "test":
            return len(self.test_data)
        elif self.mode == "val":
            return len(self.val_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.test_file)) and \
            os.path.exists(os.path.join(self.root, self.val_file))

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format(self.mode)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        return fmt_str
