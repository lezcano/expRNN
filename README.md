# Dynamic Trivializations: Cheap and Simple Manifold Constraints (Orthogonal, Positive Definite, Positive determinant...)

Code and Experiments of the papers:

"[Trivializations for Gradient-Based Optimization on Manifolds][arxivtriv]"

"[Cheap Orthogonal Constraints in Neural Networks: A Simple Parametrization of the Orthogonal and Unitary Group][arxivcheap]"

## Start putting orthogonal constraints in your code

### Orthogonal Dynamic Trivialization RNN (`dtriv`)

Just copy the main files into your code and use the class `OrthogonalRNN` included in the file `orthogonal.py`. This implmentation has [`expRNN`][arxivcheap] as a particular case, as described in the remark in [Section 7][arxivtriv]. In the models for the experiments it can be selected through the `--mode` parameter.

### Orthogonal constraints

We implement a class `Orthogonal` in the file `orthogonal.py` that can be used both as a static trivialization via the exponential map implementing `[expRNN][arxivcheap]`, or as a dynamic trivialization, implementing `[dtriv][arxivtriv]`. It can also be used as a static or a dynamic trivialization with other parametrizations of the orthogonal group, like the Cayley transform. We include the Cayley transform as an example in the experiments as well.

This layer could also be applied to other kinds of layers like CNNs, and as a helper for different kinds of decompositions in linear layers (QR, SVD, Polar, Schur...). To do this, just use the `Orthogonal` class included in the `orthogonal.py` file.

### Optimization step and general recommendations

To optimize with orthogonal constraints we need two optimizers, one for the orthogonal parameters and one for the non orthogonal. We provide a convenience function called `get_parameters` that, given a model, it returns the orthogoanl (skew-symmetric in this case) and non orthogonal parameters (cf line 137 in `1_copying.py`). In the conext of RNNs, we noticed empirically that having the lerning rate of the non-orthogonal parameters to be 10 times that of the orthogonal parameters yields the best performance.


Finally, we just have to use the second helper function `parametrization_trick` which effectively implements the idea described in [Section 4.3][arxivcheap] in a general way. To use it, just pass the model and the loss object after having computing the loss of your model. This function will return a modified loss object (cf. line 105 in `1_copying.py`).

These are the only two things that are needed to perform optimization with orthogonal constraints in your neural network.

## General manifold constraints
The framework presented in the paper "Trivializations for Gradient-Based Optimization on Manifolds" allows to put orthogonal constraints in any given manifold through the use of dynamic parametrizations. In order to create your own, just follow the instructions detailed at the beginning of the class `Parametrization` in the file `parametrization.py`.

All one has to do is to implement a class that inherits from it and implements the method `retraction`. In the [Section 6.3 and Section E][arxivtriv] we describe many different types of trivializations on different manifolds, which can be a good place to look for ideas.

We implemented a class that optimizes over the Stiefel manifold in `orthogonal.py` as an example. This is the class that we also use for the experiments.

## A note on the papers
- For the researcher who is mostly interested in the idea and how to implement it in their experiments, a reading order of the papers could be: [`Sections 1, 3.1, 3.2, 4`][arxivcheap] of the Cheap paper, and then [`Sections 1, 5, 6, E`][arxivtriv] of the Trivializations paper. 

- The NeurIPS paper "Trivializations for Gradient-Based Optimization on Manifolds" is a far reaching generalization of the paper "Cheap Orthogonal Constraints in Neural Networks". As such, some parts of the paper are more abstract, as they are more general. We recommend the interested reader to start reading the ICML paper, and just then, go for the NeurIPS paper. 

- In both papers, there are certain sections in the appendix that are more technical. These sections are not necessary for the implementation of the algorithms.

- It can be particularly userful Section E of the Trivializations paper, as it contains many possible applications of the framework in different contexts.

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

    @inproceedings{lezcano2019trivializations,
      title={Trivializations for Gradient-Based Optimization on Manifolds},
      author={Lezcano-Casado, Mario}
      booktitle={Neural Information Processing Systems},
      year={2019}
    }

    @inproceedings{lezcano2019cheap,
      title={Cheap Orthogonal Constraints in Neural Networks: A Simple Parametrization of the Orthogonal and Unitary Group},
      author={Lezcano-Casado, Mario and Mart{\'i}nez-Rubio, David},
      booktitle={International Conference on Machine Learning},
      pages={3794--3803},
      year={2019}
    }

[arxivtriv]: https://arxiv.org/abs/1909.09501
[arxivcheap]: https://arxiv.org/abs/1901.08428
