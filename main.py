import torch
from datasets import generate_data
from train import train
from evaluate import evaluate
import argparse

parser = argparse.ArgumentParser(description='MPNN')
parser.add_argument('--n_train', type=int, default=100,
                    help='Number of training examples to generate.')
parser.add_argument('--n_eval', type=int, default=100,
                    help='Number of test examples to generate.')
parser.add_argument('--problem', '-p', type=int, default=0, help='problem to train on')
parser.add_argument('--gen', action='store_true')
parser.add_argument('--epochs', '-e', type=int, default=20, help='number of epochs to train')
parser.add_argument('--n_iters', type=int, default=1, help='Number of iterations of message passing')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='L2 weight decay')
parser.add_argument('--verbosity', '-v', type=int, default=1)
parser.add_argument('--load', '-l', default=None)
parser.add_argument('--model', '-m', type=int, default=0)
parser.add_argument('--batch_size', '-b', type=int, default=100)
parser.add_argument('--hidden_dim', type=int, default=10)
parser.add_argument('--message_dim', type=int, default=10)
parser.add_argument('--vertex_state_dim', type=int, default=0)
args = parser.parse_args()

PROBLEMS = [
'arithmetic', 'has_path', 'is_connected', 'qm7', 'qm7_small'
]

MODELS = [
'mpnn', 'flat'
]

args.problem = PROBLEMS[args.problem]
args.model = MODELS[args.model]
def main():
    # Generate Data (if necessary)
    if args.gen:
        print('Generating data for the {} problem: n_train = {}, n_eval = {}'.format(args.problem, args.n_train, args.n_eval))
        generate_data('train', args.problem, args.n_train)
        generate_data('eval', args.problem, args.n_eval)

    # Train Model (if necessary)
    model = train(args)

    # Evaluate Model
    #evaluate(model, GRAPHS[args.graph], args.n_iters)


if __name__ == "__main__":
    main()
