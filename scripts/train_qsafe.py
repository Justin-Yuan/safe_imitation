import os
import yaml
from datetime import datetime
from tqdm import tqdm
import numpy as np


import torch
from torch.utils.tensorboard import SummaryWriter

from safe_il.model import QSafeNetwork
from safe_il.config import get_base_parser, AttrDict
from safe_il.utils import save_config, set_manual_seed, save_checkpoint, \
    save_command
from safe_il.data import PartialTrajectoryDataset

TASK_NAME = 'qsafe'


def train(args):
    print("----------------------------------------")
    print("Configuring")
    print("----------------------------------------\n")

    # Load predefined config files, otherwise use the parser args as config
    if args.config != '':
        with open(os.path.join(args.log_dir, 'config.yaml')) as file:
            yaml_config = yaml.load(file, Loader=yaml.Loader)
            config = AttrDict(yaml_config)
    else:
        config = args

    # Manual seeds for reproducible results
    set_manual_seed(config.seed)
    print(config)

    print("----------------------------------------")
    print("Loading Model and Data")
    print("----------------------------------------\n")

    # Check if CUDA is available, otherwise use CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print('Current device is set to: ', device)

    if (not os.path.exists(config.data_dir)):
        print('ERROR: The data directory does not exist:', config.data_dir)
        return

    # Build Model
    model = QSafeNetwork(10, 1, 256)
    model.to(device=device)

    optimizer = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=0.0005)
    start_epoch = 0

    # Load a model if resuming training
    if config.resume != '':
        checkpoint = torch.load(config.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f'Resuming Training from Epoch: {start_epoch} at Loss:{loss}')

    # Create a log directory using the current timestamp
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    config.log_dir = os.path.join(
        config.log_dir, "_".join([TASK_NAME, config.model_name, config.tag]),
        "seed{}".format(config.seed) + '_' + current_time)

    os.makedirs(config.log_dir, exist_ok=True)
    print('Logs are being written to {}'.format(config.log_dir))
    print('Use TensorBoard to look at progress!')
    summary_writer = SummaryWriter(config.log_dir)

    # Dump training information in log folder (for future reference!)
    save_config(config, config.log_dir)
    save_command(config.log_dir)

    # Load dataset
    train_dataset = PartialTrajectoryDataset(
        os.path.join(config.data_dir, 'train.npy'))
    test_dataset = PartialTrajectoryDataset(
        os.path.join(config.data_dir, 'test.npy'))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               pin_memory=True)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1,
                                              shuffle=True,
                                              pin_memory=True)

    print("----------------------------------------")
    print("Training Transporter Network")
    print("----------------------------------------\n")

    #model.train()
    pbar = tqdm(total=config.epochs)
    pbar.n = start_epoch
    pbar.refresh()
    loss_eval_best = None

    # Training Loop
    for epoch in range(start_epoch, config.epochs):
        model.train()
        loss_total = 0.0

        for i, data in enumerate(train_loader):
            t1, t2, q1, q2 = data

            t1 = t1.to(device)
            t2 = t2.to(device)
            q1 = q1.to(device)
            q2 = q2.to(device)

            assert t1.shape == t2.shape

            optimizer.zero_grad(set_to_none=True)

            pred_q1 = torch.cat(
                [torch.sum(model(item), dim=0, keepdim=True) for item in t1], dim=0)
            pred_q2 = torch.cat(
                [torch.sum(model(item), dim=0, keepdim=True) for item in t2], dim=0)

            pred_q = torch.cat([pred_q1, pred_q2], dim=1)
            rank_label = (torch.lt(q1, q2)).long()

            loss = torch.nn.CrossEntropyLoss()(pred_q, rank_label)
            loss.backward()
            optimizer.step()

            loss_total += loss.item()

        loss_total /= len(train_loader)

        # logging
        pbar.update(1)
        pbar.set_description(f'Epoch {epoch} - Loss - {loss_total:.5f}')
        summary_writer.add_scalar('loss', loss_total, epoch)
        summary_writer.flush()

        # checkpoint
        save_checkpoint(
            config.log_dir, config.model_name, {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_total
            })

        # Evaluation
        if epoch > 0 and epoch % config.eval_interval == 0:
            loss_eval_total = 0.0

            with torch.no_grad():
                for i, data in enumerate(train_loader):
                    t1, t2, q1, q2 = data

                    t1 = t1.to(device)
                    t2 = t2.to(device)
                    q1 = q1.to(device)
                    q2 = q2.to(device)

                    assert t1.shape == t2.shape

                    optimizer.zero_grad(set_to_none=True)

                    pred_q1 = torch.cat(
                        [torch.sum(model(item), dim=0, keepdim=True) for item in t1], dim=0)
                    pred_q2 = torch.cat(
                        [torch.sum(model(item), dim=0, keepdim=True) for item in t2], dim=0)

                    pred_q = torch.cat([pred_q1, pred_q2], dim=1)
                    rank_label = (torch.lt(q1, q2)).long()

                    loss = torch.nn.CrossEntropyLoss()(pred_q, rank_label)
                    loss_eval_total += loss.item()

            loss_eval_total /= len(test_loader)
            summary_writer.add_scalar('loss_eval', loss_eval_total, epoch)
            summary_writer.flush()

            if loss_eval_best is None or loss_eval_total < loss_eval_best:
                loss_eval_best = loss_eval_total
                save_checkpoint(
                    config.log_dir, "checkpoint_best", {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss_total
                    })


if __name__ == '__main__':
    parser = get_base_parser()
    args = parser.parse_args()
    train(args)
