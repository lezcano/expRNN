# ExpRNN: Cheap Orthogonal Constraints

Code and Experiments of the paper "[Cheap Orthogonal Constraints in Neural Networks: A Simple Parametrization of the Orthogonal and Unitary Group][arxiv]"

## Use expRNN in your code

Just copy the four main files into your code and start using the class `ExpRNN` included in the file `exprnn.py`.

## Add orthogonal constraints to your code

We show how to implement orthogonal constraints for non-square linear layers. This is a generalisation of the framework presented in the paper. We show an example in `orthogonal_layer.py`. This could also be applied to other kinds of layers like CNNs.

We include an example of an `Orthogonal` layer in `orthogonal_layer.py`. Once we have this, in the training code we just have to separate the parameters into two optimizers. A good rule of thumb is having the non-orthogonal learning rate 10 times larger than the orthogonal learning rate.

    lr = # Some learning rate
    optim = torch.optim.RMSprop((param for param in model.parameters()
                                 if param is not model.orth_layer.log_orthogonal_kernel), lr=lr)
    optim_orth = torch.optim.RMSprop([model.orth_layer.log_orthogonal_kernel], lr=0.1*lr)

Then, in the training loop, we just add the following line after doing `loss.backward()`:

    model.orth_layer.orthogonal_step(optim_orth)

And we are good to go! We have a layer with orthogonal constraints!

Note that if you have more than one orthogonal layer, you have to call the `orthogonal_step` function of every layer with `optim_orth` and add all the corresponding parameters to `optim_orth`. This can easily be achieved with the `apply` function present in the `torch.nn.Module` objects, defining a function that performs the step.

    def orth_step(mod, optimizer):
        if type(mod) is Orthogonal:
            mod.orthogonal_step(optimizer)

and then, applying it after `loss.backward()`:

    model.apply(lambda x: orth_step(x, optim_orth))

This trick also works to gather the `log_orthogonal_kernel` parameters for the optimizers.

## Commands to Reproduce the Experiments

    python 1_copying.py -m exact --L 1000 --hidden_size 190 --init henaff --lr 2e-4 --lr_orth 2e-5
    python 1_copying.py -m exact --L 2000 --hidden_size 190 --init henaff --lr 2e-4 --lr_orth 2e-5
    python 2_mnist.py -m exact --init cayley --hidden_size 170 --lr 7e-4 --lr_orth 7e-5
    python 2_mnist.py -m exact --init cayley --hidden_size 360 --lr 5e-4 --lr_orth 5e-5
    python 2_mnist.py -m exact --init cayley --hidden_size 512 --lr 3e-4 --lr_orth 3e-5
    python 2_mnist.py -m exact --init cayley --hidden_size 170 --lr 1e-3 --lr_orth 1e-4 --permute
    python 2_mnist.py -m exact --init cayley --hidden_size 360 --lr 7e-4 --lr_orth 7e-5 --permute
    python 2_mnist.py -m exact --init cayley --hidden_size 512 --lr 5e-4 --lr_orth 5e-5 --permute
    python 3_timit.py -m exact --init henaff --hidden_size 224 --lr 1e-3 --lr_orth 1e-4
    python 3_timit.py -m exact --init henaff --hidden_size 322 --lr 7e-4 --lr_orth 7e-5
    python 3_timit.py -m exact --init henaff --hidden_size 425 --lr 7e-4 --lr_orth 7e-5

### A note on the TIMIT experiment
The TIMIT dataset is not open, but most universities and many other institutions have access to it.

To preprocess the data of the TIMIT dataset, we used the tools provided by Wisdom on the repository:

https://github.com/stwisdom/urnn

As mentioned in the repository, first downsample the TIMIT dataset using the `downsample_audio.m` present in the `matlab` folder.

> Downsample the TIMIT dataset to 8ksamples/sec using Matlab by running downsample_audio.m from the matlab directory. Make sure you modify the paths in `downsample_audio.m` for your system.

After that, modify the file `timit_prediction.py` and add the following lines after line 529.

    np.save("lens_train.npy", lens_train)
    np.save("lens_test.npy", lens_test)
    np.save("lens_eval.npy", lens_eval)
    np.save("train_x.npy", np.transpose(train_xdata, [1, 0, 2]))
    np.save("train_z.npy", np.transpose(train_z, [1, 0, 2]))
    np.save("test_x.npy",  np.transpose(test_xdata, [1, 0, 2]))
    np.save("test_z.npy",  np.transpose(test_z, [1, 0, 2]))
    np.save("eval_x.npy",  np.transpose(eval_xdata, [1, 0, 2]))
    np.save("eval_z.npy",  np.transpose(eval_z, [1, 0, 2]))

Execute it, and feed the 9 generated files to the third experiment (cf. lines , 105-116 in `3_timit.py` in this repository).

## Cite this work

    @article{lezcano2019cheap,
      title={Cheap Orthogonal Constraints in Neural Networks: A Simple Parametrization of the Orthogonal and Unitary Group},
      author={Lezcano-Casado, Mario and Mart{\'i}nez-Rubio, David},
      journal={arXiv preprint arXiv:1901.08428},
      year={2019}
    }

[arxiv]: https://arxiv.org/abs/1901.08428
