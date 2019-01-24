import torch
import torch.nn as nn
import numpy as np
import argparse

from exprnn import ExpRNN
from approximants import exp_pade, taylor, scale_square, cayley
from initialization import henaff_init, cayley_init

parser = argparse.ArgumentParser(description='Exponential Layer Copy Task')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--hidden_size', type=int, default=190)
parser.add_argument('--T', type=int, default=4000)
parser.add_argument('--L', type=int, default=1000)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--lr_orth', type=float, default=2e-5)
parser.add_argument("--rescale", action="store_true", help='Apply scale-squaring trick')
parser.add_argument("-m", "--mode",
                    choices=["exact", "cayley", "pade", "taylor20", "lstm"],
                    default="exact",
                    type=str,
                    help="LSTM or Approximant to approximate the exponential of matrices")
parser.add_argument("--init",
                    choices=["cayley", "henaff"],
                    default="henaff",
                    type=str)

# Fix seed across experiments for reproducibility
# Same seed as that used in "Orthogonal Recurrent Neural Networks with Scaled Cayley Transform"
# https://github.com/SpartinStuff/scoRNN/blob/master/scoRNN_copying.py#L79
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(5544)
np.random.seed(5544)

args = parser.parse_args()

batch_size  = args.batch_size
hidden_size = args.hidden_size
iterations  = args.T
L           = args.L
device      = torch.device('cuda')

if args.init == "cayley":
    init = cayley_init
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


def copying_data(L, K, batch_size):
    seq = np.random.randint(1, high=9, size=(batch_size, K))
    zeros1 = np.zeros((batch_size, L))
    zeros2 = np.zeros((batch_size, K-1))
    zeros3 = np.zeros((batch_size, K+L))
    marker = 9 * np.ones((batch_size, 1))

    x = torch.LongTensor(np.concatenate((seq, zeros1, marker, zeros2), axis=1))
    y = torch.LongTensor(np.concatenate((zeros3, seq), axis=1))

    return x, y


class Model(nn.Module):
    def __init__(self, n_classes, hidden_size):
        super(Model, self).__init__()
        if args.mode == "lstm":
            self.rnn = nn.LSTMCell(n_classes + 1, hidden_size)
        else:
            self.rnn = ExpRNN(n_classes + 1, hidden_size, exponential=exp, skew_initializer=init)
        self.lin = nn.Linear(hidden_size, n_classes)
        self.loss_func = nn.CrossEntropyLoss()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.lin.weight.data, nonlinearity="relu")
        nn.init.constant_(self.lin.bias.data, 0)

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

    def loss(self, logits, y):
        return self.loss_func(logits.view(-1, 9), y.view(-1))

    def accuracy(self, logits, y):
        return torch.eq(torch.argmax(logits, dim=2), y).float().mean()

def onehot(out, input):
    out.zero_()
    in_unsq = torch.unsqueeze(input, 2)
    out.scatter_(2, in_unsq, 1)

def main():
    # --- Set data params ----------------
    n_classes = 9
    n_characters = n_classes + 1
    K = 10
    n_train = iterations *  batch_size
    n_len = L + 2 * K

    train_x, train_y = copying_data(L, K, n_train)
    train_x = train_x
    train_y = train_y

    model = Model(n_classes, hidden_size).to(device)
    model.train()

    if args.mode == "lstm":
        optim = torch.optim.RMSprop(model.parameters(), lr=args.lr)
        optim_orth = None
    else:
        optim = torch.optim.RMSprop((param for param in model.parameters()
                                     if param is not model.rnn.log_recurrent_kernel and
                                        param is not model.rnn.recurrent_kernel), lr=args.lr)
        optim_orth = torch.optim.RMSprop([model.rnn.log_recurrent_kernel], lr=args.lr_orth)

    x_onehot = torch.FloatTensor(batch_size, n_len, n_characters).to(device)

    for step in range(iterations):
        batch_x = train_x[step * batch_size : (step+1) * batch_size].to(device)
        onehot(x_onehot, batch_x)
        batch_y = train_y[step * batch_size : (step+1) * batch_size].to(device)

        logits = model(x_onehot)
        loss = model.loss(logits, batch_y)

        optim.zero_grad()
        loss.backward()
        if optim_orth:
            model.rnn.orthogonal_step(optim_orth)
        optim.step()

        with torch.no_grad():
            accuracy = model.accuracy(logits, batch_y)

        print("Iter {}: Loss= {:.6f}, Accuracy= {:.5f}".format(step, loss, accuracy))

    print("Optimization Finished!")

    model.eval()
    with torch.no_grad():
        test_x, test_y = copying_data(L, K, batch_size)
        test_x, test_y = test_x.to(device), test_y.to(device)
        onehot(x_onehot, test_x)
        logits = model(x_onehot)
        loss = model.loss(logits, test_y)
        accuracy = model.accuracy(logits, test_y)
        print("Test result: Loss= {:.6f}, Accuracy= {:.5f}".format(loss, accuracy))


if __name__ == "__main__":
    main()
