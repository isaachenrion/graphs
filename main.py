import torch
from graphs import generate
from train import train
from evaluate import evaluate
import argparse

parser = argparse.ArgumentParser(description='Phone a friend')
parser.add_argument('--task', type=int, default=0, metavar='N',
                    help='Which NPI task to run')
parser.add_argument('--num_train', type=int, default=100,
                    help='Number of training examples to generate.')
parser.add_argument('--num_test', type=int, default=100,
                    help='Number of test examples to generate.')
parser.add_argument('--graph', '-g', type=int, default=0, help='type of graph to use')
parser.add_argument('--gen', action='store_true')
parser.add_argument('--num_epochs', type=int, default=20, help='number of epochs to train (default: 10)')
parser.add_argument('--n_iters', type=int, default=1, help='Number of iterations of message passing')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
args = parser.parse_args()

GRAPHS = [
'random', 'petersen', 'cubical'
]
def main():
    #if args.task == 0:
    # Generate Data (if necessary)
    if args.gen:
        generate('train', args.num_train, graph_type=GRAPHS[args.graph])
        generate('eval', args.num_test, graph_type=GRAPHS[args.graph])

    # Train Model (if necessary)
    model = train(args.num_epochs, GRAPHS[args.graph], args.lr)

    # Evaluate Model
    #evaluate(model, GRAPHS[args.graph], args.n_iters)


if __name__ == "__main__":
    main()
