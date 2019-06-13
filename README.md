# ExpRNN: Cheap Orthogonal Constraints

Code and Experiments of the paper "[Cheap Orthogonal Constraints in Neural Networks: A Simple Parametrization of the Orthogonal and Unitary Group][arxiv]"

## Start putting orthogonal constraints in your code

### Exponential RNN (`expRNN`)

Just copy the main files into your code and use the class `ExpRNN` included in the file `exprnn.py`.

### Orthogonal constraints

We show how to implement orthogonal constraints for non-square linear layers. This is a generalisation of the framework presented in the paper. We implement it in the class `Orthogonal`. This could also be applied to other kinds of layers like CNNs, and as a helper for different kinds of decompositions in linear layers (QR, SVD, Polar, Schur...). To do this, just use the `Orthogonal` class included in the `exprnn.py` file.

### Optimization step and general recommendations

To optimize with orthogonal constraints we need two optimizers, one for the skew-symmetric parameters and one for the non orthogonal. We provide a convenience function called `get_parameters` that, given a model, it returns the skew-symmetric parameters and the non orthogonal parameters (cf. line 113 in `1_copying.py`). In the conext of RNNs, we noticed empirically that having the lerning rate of the non-orthogonal parameters to be 10 times that of the skew-symmetric parameters yields the best performance.

Finally, to execute the gradient step, we provide another convenience function called `orthogonal_step` which, given a model and the orthogonal optimizer, it performs a gradient step and updates the orthogonal matrix (cf. line 134 in `1_copying.py`). This function effectively implements the ideas in section 4.3 in the paper.

These are the only two things that are needed to perform optimization with orthogonal constraints in your neural network.

## Commands to Reproduce the Experiments

    python 1_copying.py -m exprnn --L 1000 --hidden_size 190 --init henaff --lr 2e-4 --lr_orth 2e-5
    python 1_copying.py -m exprnn --L 2000 --hidden_size 190 --init henaff --lr 2e-4 --lr_orth 2e-5
    python 2_mnist.py -m exprnn --init cayley --hidden_size 170 --lr 7e-4 --lr_orth 7e-5
    python 2_mnist.py -m exprnn --init cayley --hidden_size 360 --lr 5e-4 --lr_orth 5e-5
    python 2_mnist.py -m exprnn --init cayley --hidden_size 512 --lr 3e-4 --lr_orth 3e-5
    python 2_mnist.py -m exprnn --init cayley --hidden_size 170 --lr 1e-3 --lr_orth 1e-4 --permute
    python 2_mnist.py -m exprnn --init cayley --hidden_size 360 --lr 7e-4 --lr_orth 7e-5 --permute
    python 2_mnist.py -m exprnn --init cayley --hidden_size 512 --lr 5e-4 --lr_orth 5e-5 --permute
    python 3_timit.py -m exprnn --init henaff --hidden_size 224 --lr 1e-3 --lr_orth 1e-4
    python 3_timit.py -m exprnn --init henaff --hidden_size 322 --lr 7e-4 --lr_orth 7e-5
    python 3_timit.py -m exprnn --init henaff --hidden_size 425 --lr 7e-4 --lr_orth 7e-5


## A note on the TIMIT experiment
The TIMIT dataset is not open, but most universities and many other institutions have access to it.

To preprocess the data of the TIMIT dataset, we used the tools provided by Wisdom on the repository:

https://github.com/stwisdom/urnn

As mentioned in the repository, first downsample the TIMIT dataset using the `downsample_audio.m` present in the `matlab` folder.

> Downsample the TIMIT dataset to 8ksamples/sec using Matlab by running downsample_audio.m from the matlab directory. Make sure you modify the paths in `downsample_audio.m` for your system.

Create a `timit_data` folder to store all the files.

After that, modify the file `timit_prediction.py` and add the following lines after line 529.

    np.save("timit_data/lens_train.npy", lens_train)
    np.save("timit_data/lens_test.npy", lens_test)
    np.save("timit_data/lens_eval.npy", lens_eval)
    np.save("timit_data/train_x.npy", np.transpose(train_xdata, [1, 0, 2]))
    np.save("timit_data/train_z.npy", np.transpose(train_z, [1, 0, 2]))
    np.save("timit_data/test_x.npy",  np.transpose(test_xdata, [1, 0, 2]))
    np.save("timit_data/test_z.npy",  np.transpose(test_z, [1, 0, 2]))
    np.save("timit_data/eval_x.npy",  np.transpose(eval_xdata, [1, 0, 2]))
    np.save("timit_data/eval_z.npy",  np.transpose(eval_z, [1, 0, 2]))

Run this script to save the dataset in a format that can be loaded by the TIMIT dataset loader

    import numpy as np
    import torch

    train_x = torch.tensor(np.load('timit_data/train_x.npy'))
    train_y = torch.tensor(np.load('timit_data/train_z.npy'))
    lens_train = torch.tensor(np.load("timit_data/lens_train.npy"), dtype=torch.long)

    test_x = torch.tensor(np.load('timit_data/test_x.npy'))
    test_y = torch.tensor(np.load('timit_data/test_z.npy'))
    lens_test = torch.tensor(np.load("timit_data/lens_test.npy"), dtype=torch.long)

    val_x = torch.tensor(np.load('timit_data/eval_x.npy'))
    val_y = torch.tensor(np.load('timit_data/eval_z.npy'))
    lens_val = torch.tensor(np.load("timit_data/lens_eval.npy"), dtype=torch.long)

    training_set = (train_x, train_y, lens_train)
    test_set = (test_x, test_y, lens_test)
    val_set = (val_x, val_y, lens_val)
    with open("timit_data/training.pt", 'wb') as f:
        torch.save(training_set, f)
    with open("timit_data/test.pt", 'wb') as f:
        torch.save(test_set, f)
    with open("timit_data/val.pt", 'wb') as f:
        torch.save(val_set, f)

## Cite this work

    @inproceedings{lezcano2019cheap,
      title={Cheap Orthogonal Constraints in Neural Networks: A Simple Parametrization of the Orthogonal and Unitary Group},
      author={Lezcano-Casado, Mario and Mart{\'i}nez-Rubio, David},
      booktitle={International Conference on Machine Learning},
      pages={3794--3803},
      year={2019}
    }

[arxiv]: https://arxiv.org/abs/1901.08428
