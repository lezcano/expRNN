import torch
import torch.nn as nn
import numpy as np
import argparse
from torchvision import datasets, transforms

from exprnn import ExpRNN
from approximants import exp_pade, taylor, scale_square, cayley
from initialization import henaff_init, cayley_init

parser = argparse.ArgumentParser(description='Exponential Layer MNIST Task')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--hidden_size', type=int, default=170)
parser.add_argument('--epochs', type=int, default=70)
parser.add_argument('--lr', type=float, default=7e-4)
parser.add_argument('--lr_orth', type=float, default=7e-5)
parser.add_argument("--permute", action="store_true")
parser.add_argument("--rescale", action="store_true")
parser.add_argument("-m", "--mode",
                    choices=["exact", "cayley", "pade", "taylor20", "lstm"],
                    default="exact",
                    type=str,
                    help="LSTM or Approximant to approximate the exponential of matrices")
parser.add_argument("--init",
                    choices=["cayley", "henaff"],
                    default="cayley",
                    type=str)


args = parser.parse_args()

# Fix seed across experiments
# Same seed as that used in "Orthogonal Recurrent Neural Networks with Scaled Cayley Transform"
# https://github.com/SpartinStuff/scoRNN/blob/master/scoRNN_copying.py#L79
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(5544)
np.random.seed(5544)

n_classes   = 10
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

if args.permute:
    permute = np.random.RandomState(92916)
    permutation = torch.LongTensor(permute.permutation(784))


class Model(nn.Module):
    def __init__(self, hidden_size):
        super(Model, self).__init__()
        if args.mode == "lstm":
            self.rnn = nn.LSTMCell(1, hidden_size)
        else:
            self.rnn = ExpRNN(1, hidden_size, exponential=exp, skew_initializer=init)
        self.lin = nn.Linear(hidden_size, n_classes)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, inputs):
        if args.permute:
            inputs = inputs[:, permutation]
        h = None
        for input in torch.unbind(inputs, dim=1):
            h = self.rnn(input.unsqueeze(dim=1), h)
        return self.lin(h)

    def loss(self, logits, y):
        return self.loss_func(logits, y)

    def correct(self, logits, y):
        return torch.eq(torch.argmax(logits, dim=1), y).float().sum()


def main():
    # Load data
    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./mnist', train=True, download=True, transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./mnist', train=False, transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True, **kwargs)

    # Model and optimizers
    model = Model(hidden_size).to(device)
    model.train()

    if args.mode == "lstm":
        optim = torch.optim.RMSprop(model.parameters(), lr=args.lr)
        optim_orth = None
    else:
        optim = torch.optim.RMSprop((param for param in model.parameters()
                                     if param is not model.rnn.log_recurrent_kernel and
                                        param is not model.rnn.recurrent_kernel), lr=args.lr)
        optim_orth = torch.optim.RMSprop([model.rnn.log_recurrent_kernel], lr=args.lr_orth)

    best_test_acc = 0.
    for epoch in range(epochs):
        processed = 0
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device).view(-1, 784), batch_y.to(device)

            logits = model(batch_x)
            loss = model.loss(logits, batch_y)

            optim.zero_grad()
            loss.backward()
            if optim_orth:
                model.rnn.orthogonal_step(optim_orth)
            optim.step()

            with torch.no_grad():
                correct = model.correct(logits, batch_y)

            processed += len(batch_x)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.6f}%'.format(
                epoch, processed, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), correct/len(batch_x)))

        model.eval()
        with torch.no_grad():
            test_loss = 0.
            correct = 0.
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device).view(-1, 784), batch_y.to(device)
                logits = model(batch_x)
                test_loss += model.loss(logits, batch_y).float()
                correct += model.correct(logits, batch_y).float()

        test_loss /= len(test_loader)
        test_acc = correct / len(test_loader.dataset)
        best_test_acc = max(test_acc, best_test_acc)
        print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Best Accuracy: {:.3f}\n"
                .format(test_loss, correct, len(test_loader.dataset), 100 * test_acc, best_test_acc))

        model.train()


if __name__ == "__main__":
    main()
