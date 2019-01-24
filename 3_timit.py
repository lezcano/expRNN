import torch
import torch.nn as nn
import numpy as np
import argparse

from exprnn import ExpRNN
from approximants import exp_pade, taylor, scale_square, cayley
from initialization import henaff_init, cayley_init

parser = argparse.ArgumentParser(description='Exponential Layer TIMIT Task')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--hidden_size', type=int, default=224)
parser.add_argument('--epochs', type=int, default=1200)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_orth', type=float, default=1e-4)
parser.add_argument("--rescale", action="store_true")
parser.add_argument("-m", "--mode",
                    choices=["exact", "cayley", "pade", "taylor20", "lstm"],
                    default="exact",
                    type=str,
                    help="LSTM or Approximant to approximate the exponential of matrices")
parser.add_argument("--init",
                    choices=["cayley", "henaff"],
                    default="henaff",
                    type=str)

args = parser.parse_args()

# Fix seed across experiments
# Same seed as that used in "Orthogonal Recurrent Neural Networks with Scaled Cayley Transform"
# https://github.com/SpartinStuff/scoRNN/blob/master/scoRNN_copying.py#L79

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(5544)
np.random.seed(5544)

n_input     = 129
n_classes   = 129
batch_size  = args.batch_size
hidden_size = args.hidden_size
epochs      = args.epochs
device      = torch.device('cuda')

if args.init == "cayley":
    init =  cayley_init
elif args.init == "henaff":
    init = henaff_init

if args.mode == "cayley":
    exp_func = cayley
elif args.mode == "pade":
    exp_func = exp_pade
elif args.mode == "taylor20":
    exp_func = lambda X: taylor(X, 20)

if args.mode != "lstm":
    if args.mode == "exact":
        # The exact implementation already implements a more advanced form of scale-squaring trick
        exp = "exact"
    else:
        if args.rescale:
            exp = lambda X: scale_square(X, exp_func)
        else:
            exp = exp_func


def masked_loss(lossfunc, logits, y, lens):
    """ Computes the loss of the first `lens` items in the batches """
    mask = torch.zeros_like(logits, dtype=torch.uint8)
    for i, l in enumerate(lens):
        mask[i, :l, :] = 1
    logits_masked = torch.masked_select(logits, mask)
    y_masked = torch.masked_select(y, mask)
    return lossfunc(logits_masked, y_masked)


class Model(nn.Module):
    def __init__(self, hidden_size):
        super(Model, self).__init__()
        if args.mode == "lstm":
            self.rnn = nn.LSTMCell(n_input, hidden_size)
        else:
            self.rnn = ExpRNN(n_input, hidden_size, exponential=exp, skew_initializer=init)
        self.lin = nn.Linear(hidden_size, n_classes)
        self.loss_func = nn.MSELoss()

    def forward(self, inputs):
        h = None
        out = []
        for input in torch.unbind(inputs, dim=1):
            h = self.rnn(input, h)
            if isinstance(h, tuple):
                out_rnn = h[0]
            else:
                out_rnn = h
            out.append(self.lin(out_rnn))
        return torch.stack(out, dim=1)

    def loss(self, logits, y, len_batch):
        return masked_loss(self.loss_func, logits, y, len_batch)


def main():
    # Load data
    train_x = np.load('timit_data/train_x.npy')
    train_y = np.load('timit_data/train_z.npy')
    lens_train = np.load("timit_data/lens_train.npy")

    test_x = torch.tensor(np.load('timit_data/test_x.npy'), device=device)
    test_y = torch.tensor(np.load('timit_data/test_z.npy'), device=device)
    lens_test = torch.LongTensor(np.load("timit_data/lens_test.npy"), device=device)

    val_x = torch.tensor(np.load('timit_data/eval_x.npy'), device=device)
    val_y = torch.tensor(np.load('timit_data/eval_z.npy'), device=device)
    lens_val = torch.LongTensor(np.load("timit_data/lens_eval.npy"), device=device)

    # Shuffle data
    n_train = train_x.shape[0]
    p = np.random.permutation(n_train)
    train_x = train_x[p]
    train_y = train_y[p]
    lens_train = lens_train[p]

    # Batching
    x_batches = torch.tensor(np.array_split(train_x, int(n_train / batch_size)))
    y_batches = torch.tensor(np.array_split(train_y, int(n_train / batch_size)))
    lens_batches = torch.LongTensor(np.array_split(lens_train, int(n_train / batch_size)))

    # Model and optimizers
    model = Model(hidden_size).to(device)
    model.train()

    if args.mode == "lstm":
        optim = torch.optim.Adam(model.parameters(), lr=args.lr)
        optim_orth = None
    else:
        optim = torch.optim.Adam((param for param in model.parameters()
                                  if param is not model.rnn.log_recurrent_kernel and
                                     param is not model.rnn.recurrent_kernel), lr=args.lr)
        optim_orth = torch.optim.RMSprop([model.rnn.log_recurrent_kernel], lr=args.lr_orth)

    best_test = 1e7
    best_validation = 1e7
    for epoch in range(epochs):
        processed = 0
        step = 1
        for batch_idx, (batch_x, batch_y, len_batch) in enumerate(zip(x_batches, y_batches, lens_batches)):
            batch_x, batch_y, len_batch = batch_x.to(device), batch_y.to(device), len_batch.to(device)

            logits = model(batch_x)
            loss = model.loss(logits, batch_y, len_batch)

            optim.zero_grad()
            loss.backward()
            if optim_orth:
                model.rnn.orthogonal_step(optim_orth)
            optim.step()

            processed += len(batch_x)
            step += 1

            print("Epoch {} [{}/{} ({:.0f}%)]\tLoss: {:.2f} "
                  .format(epoch, processed, len(train_x), 100. * processed / len(train_x), loss))

        model.eval()
        with torch.no_grad():
            logits_val = model(val_x)
            loss_val = model.loss(logits_val, val_y, lens_val)
            logits_test = model(test_x)
            loss_test = model.loss(logits_test, test_y, lens_test)
            if loss_val < best_validation:
                best_validation = loss_val
                best_test = loss_test
            print()
            print("Val:  Loss: {:.2f}\tBest: {:.2f}".format(loss_val, best_validation))
            print("Test: Loss: {:.2f}\tBest: {:.2f}".format(loss_test, best_test))
            print()
        model.train()


if __name__ == "__main__":
    main()
