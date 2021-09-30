import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from safe_il.config import get_plot_parser
from safe_il.model import QSafeNetwork


def plot(args):
    # Check if CUDA is available, otherwise use CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Build Model
    model = QSafeNetwork(10, 1, 256)
    checkpoint = torch.load(os.path.join(args.log_dir + 'model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device=device)


    states = []
    x_segments = 100
    y_segments = 100

    # TODO(mustafa): Use meshgrid?
    for x in np.linspace(0, 1, x_segments):
        for y in np.linspace(0, 1, y_segments):
            states.append([x, y, 0.9, 0.9, 0.3, 0.2, 0.8, 0.2, 0.4, 0.7])

    # TODO(mustafa): Should use info from environment for states ^

    pred = []
    with torch.no_grad():
        for state in states:
            state = torch.Tensor(state).to(device)
            out = model(state)
            pred.append(out.detach().cpu().numpy())

    pred = np.array(pred)
    pred = pred.reshape(y_segments, x_segments)

    print(pred)
    plt.gca().add_patch(
        Circle((30, 20), radius=2, color='r')
    )
    plt.gca().add_patch(
        Circle((80, 20), 2,  color='r')
    )
    plt.gca().add_patch(
        Circle((40, 70), 2,  color='r')
    )
    plt.gca().add_patch(
        Circle((90, 90), 2,  color='g')
    )
    plt.imshow(pred.T)
    filepath = os.path.join(args.log_dir, args.filename)
    plt.savefig(filepath, bbox_inches='tight')


if __name__ == '__main__':
    parser = get_plot_parser()
    args = parser.parse_args()
    plot(args)