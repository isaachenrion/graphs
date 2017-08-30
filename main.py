import torch
from datasets import generate_data
from train import train
import argparse

parser = argparse.ArgumentParser(description='MPNN')
parser.add_argument('--n_train', type=int, default=100,
                    help='Number of training examples to generate.')
parser.add_argument('--n_eval', type=int, default=100,
                    help='Number of test examples to generate.')
parser.add_argument('--problem', '-p', type=int, default=0, help='problem to train on')
parser.add_argument('--gen', action='store_true')
parser.add_argument('--epochs', '-e', type=int, default=100, help='number of epochs to train')
parser.add_argument('--n_iters', type=int, default=1, help='Number of iterations of message passing')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0.00005, help='L2 weight decay')
parser.add_argument('--verbosity', '-v', type=int, default=2)
parser.add_argument('--load', '-l', default=None)
parser.add_argument('--model', '-m', type=int, default=0)
parser.add_argument('--batch_size', '-b', type=int, default=100)

parser.add_argument('--hidden_dim', type=int, default=100)
parser.add_argument('--message_dim', type=int, default=100)
parser.add_argument('--vertex_state_dim', type=int, default=0)

parser.add_argument('--readout', default=None)
parser.add_argument('--message', default=None)
parser.add_argument('--vertex_update', default=None)
parser.add_argument('--embedding', default=None)

parser.add_argument('--debug', action='store_true')

args = parser.parse_args()

PROBLEMS = [
    'qm7', # 0
    'qm7_small', # 1
    'qm7b', # 2
    'qm7b_small', # 3
    'arithmetic', # 4
    'has_path', # 5
    'is_connected', # 6
    'simple', # 7
]
args.problem = PROBLEMS[args.problem]

MODELS = [
'mpnn', 'flat', 'vcn'
]
args.model = MODELS[args.model]
if args.model == 'vcn':
    args.readout = 'selected_vertices'
    args.message = 'fully_connected'
    args.vertex_update = 'gru'
    args.embedding = 'constant'
elif args.model == 'mpnn':
    args.readout = 'fully_connected'
    args.message = 'fully_connected'
    args.vertex_update = 'gru'
    args.embedding = 'constant'
elif args.model == 'flat':
    args.readout = None
    args.message = None
    args.vertex_update = None
    args.embedding = None

def main():
    if args.gen:
        print('Generating data for the {} problem'.format(args.problem))
        generate_data('train', args.problem, args.n_train)
        generate_data('eval', args.problem, args.n_eval)

    # Train Model (if necessary)
    model = train(args)

if __name__ == "__main__":
    main()
