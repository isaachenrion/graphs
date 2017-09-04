import torch
from datasets import generate_data
from train import train
import argparse
import os
import linecache
import tracemalloc
parser = argparse.ArgumentParser(description='MPNN')

# data args
parser.add_argument('--n_train', type=int, default=100, help='Number of training examples to generate.')
parser.add_argument('--n_eval', type=int, default=100, help='Number of test examples to generate.')
parser.add_argument('--problem', '-t', type=int, default=0, help='task to train on')
parser.add_argument('--gen', action='store_true', help='generate the data')

# miscellaneous args
parser.add_argument('--gpu', type=int, default=0, help='Device to use (GPU)')
parser.add_argument('--verbosity', '-v', type=int, default=2)
parser.add_argument('--debug', action='store_true')

# training args
parser.add_argument('--epochs', '-e', type=int, default=100, help='number of epochs to train')
parser.add_argument('--batch_size', '-b', type=int, default=100, help='batch size for training')
parser.add_argument('--randomize_nodes', '-r', action='store_true', help='Randomize the ordering of the nodes')
parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0.00005, help='L2 weight decay')

# model args
parser.add_argument('--load', '-l', default=None)
parser.add_argument('--model', '-m', type=int, default=0)

# model dimension args
parser.add_argument('--hidden_dim', type=int, default=100)
parser.add_argument('--message_dim', type=int, default=100)
parser.add_argument('--vertex_state_dim', type=int, default=0)

# mpnn args
parser.add_argument('--n_iters', type=int, default=1, help='Number of iterations of message passing')
parser.add_argument('--parallelism', '-p', type=int, default=1, help='MPNN parallelism level (0, 1)')

parser.add_argument('--readout', default=None)
parser.add_argument('--message', default=None)
parser.add_argument('--vertex_update', default=None)
parser.add_argument('--embedding', default=None)


args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

if args.debug:
    args.problem = 1
    args.weight_decay = 0.
    args.hidden_dim = 7
    #args.n_iters = 2
    args.message_dim = 11
    args.batch_size = 23

PROBLEMS = [
    'qm7_edge_representation',
    'qm7_edge_representation_small',
    'qm7_vertex_representation',
    'qm7_vertex_representation_small',
    'qm7b',
    'qm7b_small',
    'arithmetic',
    'has_path',
    'is_connected',
    'simple',
]
args.problem = PROBLEMS[args.problem]

MODELS = [
'mpnn', 'flat', 'vcn', 'mpnn_set', 'dtnn'
]
args.model = MODELS[args.model]
if args.model == 'vcn':
    args.readout = 'vcn'
    args.message = 'fully_connected'
    args.vertex_update = 'gru'
    args.embedding = 'constant'
elif args.model == 'mpnn':
    args.readout = 'fully_connected'
    args.message = 'constant' # 'fully_connected'
    args.vertex_update = 'gru'
    args.embedding = 'constant'
elif args.model == 'mpnn_set':
    args.readout = 'set'
    args.message = 'constant' # 'fully_connected'
    args.vertex_update = 'gru'
    args.embedding = 'constant'
elif args.model == 'dtnn':
    args.readout = 'dtnn'
    args.message = 'dtnn' # 'fully_connected'
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

def display_top(snapshot, key_type='lineno', limit=10):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))

if __name__ == "__main__":
    tracemalloc.start(25)

    main()

    snapshot = tracemalloc.take_snapshot()
    display_top(snapshot)
