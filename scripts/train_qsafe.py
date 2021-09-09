import os
import torch

from safe_il.config import get_base_parser
from safe_il.utils import set_manual_seed, save_checkpoint, save_command


def train(config):
    print("----------------------------------------")
    print("Configuring")
    print("----------------------------------------\n")

    # Manual seeds for reproducible results
    set_manual_seed(config.seed)
    print(config)

    print("----------------------------------------")
    print("Loading Model and Data")
    print("----------------------------------------\n")

    # Check if CUDA is available, otherwise use CPU
    # if torch.cuda.is_available():
    #     device = torch.device('cuda')
    # else:
    #     device = torch.device('cpu')
    device = torch.device('cpu')

    print('Current device is set to: ', device)

    if (not os.path.exists(config.data_dir)):
        print('The data directory does not exist:', config.data_dir)
        return

    # Build Model
    agent = make_agent(config)
    agent.to(device=device)

    optimizer = torch.optim.Adam(agent.model.parameters(), 1e-3)
    start_epoch = 0

    # Load a model if resuming training
    if config.resume != '':
        checkpoint = torch.load(config.resume)
        agent.load_state_dict(checkpoint['model_state_dict'])
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

    print("----------------------------------------")
    print("Training Transporter Network")
    print("----------------------------------------\n")

    pbar = tqdm(total=config.num_epochs)
    pbar.n = start_epoch
    pbar.refresh()

    # load dataset
    dataset = load_data(config, config.data_dir)
    dataset_train, dataset_test = split_train_test(dataset,
                                                   train_ratio=TRAIN_RATIO)
    dataset_train = ReachTargetDataset(data_list=dataset_train)
    loader = torch.utils.data.DataLoader(dataset_train,
                                         batch_size=config.batch_size,
                                         shuffle=True)
    dataset_test = ReachTargetDataset(data_list=dataset_test)
    loader_test = torch.utils.data.DataLoader(dataset_test,
                                              batch_size=config.batch_size)

    # trianing loop
    loss_eval_best = None
    if config.eval_when_train:
        # make env
        env, task = make_env(headless=True, full_obs=False)

    for epoch in range(start_epoch, config.num_epochs):
        agent.train()
        loss_total = 0.0

        for i, (features, labels) in enumerate(loader):
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            out = agent.output(features)
            loss = torch.nn.functional.mse_loss(out, labels)
            loss_total += loss.item()

            loss.backward()
            optimizer.step()

        loss_total /= len(loader)

        # logging
        pbar.update(1)
        pbar.set_description(f'Epoch {epoch} - Loss - {loss_total:.5f}')
        summary_writer.add_scalar('loss', loss_total, epoch)
        summary_writer.flush()

        # checkpoint
        save_checkpoint(
            config.log_dir, "checkpoint", {
                'epoch': epoch,
                'model_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_total
            })

        # evaluation
        agent.eval()
        loss_eval_total = 0.0

        for features, labels in loader_test:
            features = features.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                out = agent.output(features)
            loss_eval = torch.nn.functional.mse_loss(out, labels)
            loss_eval_total += loss_eval.item()

        loss_eval_total /= len(loader_test)
        summary_writer.add_scalar('loss_eval', loss_eval_total, epoch)
        summary_writer.flush()

        if loss_eval_best is None or loss_eval_total < loss_eval_best:
            loss_eval_best = loss_eval_total
            save_checkpoint(
                config.log_dir, "checkpoint_best", {
                    'epoch': epoch,
                    'model_state_dict': agent.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_total
                })

        # evaluation rollouts
        if config.eval_when_train and epoch > 0 and epoch % config.eval_when_train_freq == 0:
            results = collect_rollouts(
                config,
                task,
                agent,
                device=device,
                max_episode_length=config.max_episode_length,
                batch_size=config.eval_batch_size,
                render=config.render,
                vid_dir=config.log_dir)

            total_lengths = results["total_lengths"]
            total_rewards = results["total_rewards"]
            total_costs = results["total_costs"]

            summary_writer.add_scalar('loss_eval/total_lengths',
                                      np.mean(total_lengths), epoch)
            summary_writer.add_scalar('loss_eval/total_rewards',
                                      np.mean(total_rewards), epoch)
            summary_writer.add_scalar('loss_eval/total_costs',
                                      np.mean(total_costs), epoch)
            summary_writer.flush()

    # clean up
    if config.eval_when_train:
        env.shutdown()


if __name__ == '__main__':
    parser = get_base_parser()
    args = parser.parse_args()
    train(args)