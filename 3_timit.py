import torch
import torch.nn as nn
import numpy as np
import argparse
import datetime

from exprnn import ExpRNN, get_parameters, orthogonal_step
from initialization import henaff_init, cayley_init
from timit_loader import TIMIT

parser = argparse.ArgumentParser(description='Exponential Layer TIMIT Task')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--hidden_size', type=int, default=224)
parser.add_argument('--epochs', type=int, default=1200)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_orth', type=float, default=1e-4)
parser.add_argument("-m", "--mode",
                    choices=["exprnn", "lstm"],
                    default="exprnn",
                    type=str)
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


def masked_loss(lossfunc, logits, y, lens):
    """ Computes the loss of the first `lens` items in the batches """
    mask = torch.zeros_like(logits, dtype=torch.uint8)
    for i, l in enumerate(lens):
        mask[i, :l, :] = 1
    logits_masked = torch.masked_select(logits, mask)
    y_masked = torch.masked_select(y, mask)
    return lossfunc(logits_masked, y_masked)


class Model(torch.jit.ScriptModule):
    def __init__(self, hidden_size):
        super(Model, self).__init__()
        if args.mode == "lstm":
            self.rnn = nn.LSTMCell(n_input, hidden_size)
        else:
            self.rnn = ExpRNN(n_input, hidden_size, skew_initializer=init)
        self.lin = nn.Linear(hidden_size, n_classes)
        self.loss_func = nn.MSELoss()

    @torch.jit.script_method
    def forward(self, inputs):
        state = self.rnn.default_hidden(inputs[:, 0, ...])
        outputs = torch.jit.annotate(List[Tensor], [])
        for input in torch.unbind(inputs, dim=1):
            out_rnn, state = self.rnn(input, state)
            outputs += [self.lin(out_rnn)]
        return torch.stack(outputs, dim=1)

    def loss(self, logits, y, len_batch):
        return masked_loss(self.loss_func, logits, y, len_batch)


def main():
    # Load data
    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        TIMIT('./timit_data', mode="train"),
        batch_size=batch_size, shuffle=True, **kwargs)
    # Load test and val in one big batch
    test_loader = torch.utils.data.DataLoader(
        TIMIT('./timit_data', mode="test"),
        batch_size=400, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        TIMIT('./timit_data', mode="val"),
        batch_size=192, shuffle=True, **kwargs)


    # Model and optimizers
    model = Model(hidden_size).to(device)
    model.train()

    if args.mode == "lstm":
        optim = torch.optim.RMSprop(model.parameters(), lr=args.lr)
        optim_orth = None
    else:
        non_orth_params, log_orth_params = get_parameters(model)
        optim = torch.optim.Adam(non_orth_params, args.lr)
        optim_orth = torch.optim.RMSprop(log_orth_params, lr=args.lr_orth)

    best_test = 1e7
    best_validation = 1e7

    for epoch in range(epochs):
        init_time = datetime.datetime.now()
        processed = 0
        step = 1
        for batch_idx, (batch_x, batch_y, len_batch) in enumerate(train_loader):
            batch_x, batch_y, len_batch = batch_x.to(device), batch_y.to(device), len_batch.to(device)

            logits = model(batch_x)
            loss = model.loss(logits, batch_y, len_batch)

            # Zeroing out the optim_orth is not really necessary, but we do it for consistency
            if optim_orth:
                optim_orth.zero_grad()

            optim.zero_grad()

            loss.backward()

            if optim_orth:
                model.apply(orthogonal_step(optim_orth))
            optim.step()

            processed += len(batch_x)
            step += 1

            print("Epoch {} [{}/{} ({:.0f}%)]\tLoss: {:.2f} "
                  .format(epoch, processed, len(train_loader.dataset),
                      100. * processed / len(train_loader.dataset), loss))

        model.eval()
        with torch.no_grad():
            # There's just one batch for test and validation
            for batch_x, batch_y, len_batch in test_loader:
                batch_x, batch_y, len_batch = batch_x.to(device), batch_y.to(device), len_batch.to(device)
                logits = model(batch_x)
                loss_test = model.loss(logits, batch_y, len_batch)

            for batch_x, batch_y, len_batch in val_loader:
                batch_x, batch_y, len_batch = batch_x.to(device), batch_y.to(device), len_batch.to(device)
                logits = model(batch_x)
                loss_val = model.loss(logits, batch_y, len_batch)

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
