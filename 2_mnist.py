import torch
import torch.nn as nn
import numpy as np
import argparse
import sys
from torchvision import datasets, transforms

from parametrization import parametrization_trick, get_parameters
from orthogonal import OrthogonalRNN
from trivializations import cayley_map, expm_skew
from initialization import henaff_init_, cayley_init_


parser = argparse.ArgumentParser(description='Exponential Layer MNIST Task')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--hidden_size', type=int, default=170)
parser.add_argument('--epochs', type=int, default=70)
parser.add_argument('--lr', type=float, default=7e-4)
parser.add_argument('--lr_orth', type=float, default=7e-5)
parser.add_argument("--permute", action="store_true")
parser.add_argument("-m", "--mode",
                    choices=["exprnn", "dtriv", "cayley", "lstm"],
                    default="dtriv",
                    type=str)
parser.add_argument('--K', type=str, default="100", help='The K parameter in the dtriv algorithm. It should be a positive integer or "infty".')
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
    init =  cayley_init_
elif args.init == "henaff":
    init = henaff_init_

if args.K != "infty":
    args.K = int(args.K)
if args.mode == "exprnn":
    mode = "static"
    param = expm_skew
elif args.mode == "dtriv":
    # We use 100 as the default to project back to the manifold.
    # This parameter does not really affect the convergence of the algorithms, even for K=1
    mode = ("dynamic", args.K, 100)
    param = expm_skew
elif args.mode == "cayley":
    mode = "static"
    param = cayley_map


class Model(nn.Module):
    def __init__(self, hidden_size, permute):
        super(Model, self).__init__()
        self.permute = permute
        permute = np.random.RandomState(92916)
        self.register_buffer("permutation", torch.LongTensor(permute.permutation(784)))
        if args.mode == "lstm":
            self.rnn = nn.LSTMCell(1, hidden_size)
        else:
            self.rnn = OrthogonalRNN(1, hidden_size, skew_initializer=init, mode=mode, param=param)

        self.lin = nn.Linear(hidden_size, n_classes)
        self.loss_func = nn.CrossEntropyLoss()


    def forward(self, inputs):
        if self.permute:
            inputs = inputs[:, self.permutation]

        if isinstance(self.rnn, OrthogonalRNN):
            state = self.rnn.default_hidden(inputs[:, 0, ...])
        else:
            state = (torch.zeros((inputs.size(0), self.hidden_size), device=inputs.device),
                     torch.zeros((inputs.size(0), self.hidden_size), device=inputs.device))
        for input in torch.unbind(inputs, dim=1):
            out_rnn, state = self.rnn(input.unsqueeze(dim=1), state)
            if isinstance(self.rnn, nn.LSTMCell):
                state = (out_rnn, state)
        return self.lin(state)

    def loss(self, logits, y):
        l = self.loss_func(logits, y)
        if isinstance(self.rnn, OrthogonalRNN):
            return parametrization_trick(model=self, loss=l)
        else:
            return l

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
    model = Model(hidden_size, args.permute).to(device)
    model.train()

    if args.mode == "lstm":
        optim = torch.optim.RMSprop(model.parameters(), lr=args.lr)
        optim_orth = None
    else:
        non_orth_params, log_orth_params = get_parameters(model)
        optim = torch.optim.RMSprop(non_orth_params, args.lr)
        optim_orth = torch.optim.RMSprop(log_orth_params, lr=args.lr_orth)

    best_test_acc = 0.
    for epoch in range(epochs):
        processed = 0
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device).view(-1, 784), batch_y.to(device)

            logits = model(batch_x)
            loss = model.loss(logits, batch_y)

            optim.zero_grad()
            # Zeroing out the optim_orth is not really necessary, but we do it for consistency
            if optim_orth:
                optim_orth.zero_grad()

            loss.backward()

            optim.step()
            if optim_orth:
                optim_orth.step()

            with torch.no_grad():
                correct = model.correct(logits, batch_y)

            processed += len(batch_x)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.2f}%\tBest: {:.2f}%'.format(
                epoch, processed, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), 100 * correct/len(batch_x), best_test_acc))


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
        test_acc = 100 * correct / len(test_loader.dataset)
        best_test_acc = max(test_acc, best_test_acc)
        print()
        print("Test set: Average loss: {:.4f}, Accuracy: {:.2f}%, Best Accuracy: {:.2f}%"
                .format(test_loss, test_acc, best_test_acc))
        print()

        model.train()


if __name__ == "__main__":
    main()
