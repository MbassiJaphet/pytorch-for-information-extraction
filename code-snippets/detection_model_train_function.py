# import training and evaluation functions
from modules.detection.scripts.engine import train_one_epoch, evaluate

def train_detection_model(model, num_epochs=10, loaders=None, checkpoint=None, checkpoint_path=None,
                optimizer= None, lr_scheduler= None, print_freq=1, device=torch.device('cuda')):

    if checkpoint is None: start_epoch = 1
    else:
        print('Resuming training from checkpoint...')
        start_epoch = checkpoint['epoch'] + 1

    model.to(device) # Move model to cpu or cuda device

    time_train = time.time()
    for epoch in range(start_epoch, num_epochs + 1):
        time_epoch = time.time()
        # train for one epoch, printing every '{print_freq}' iterations
        train_one_epoch(model, optimizer, loaders['train'], device, epoch, print_freq=print_freq)
        # update the learning rate
        lr_scheduler.step()
        # evaluate model on the validation dataset
        evaluate(model, loaders['valid'], device=device)

        time_epoch_elapsed = time.time() - time_epoch
        time_train_elapsed = time.time() - time_train
        print('Epoch: {}\tEpoch Time: {:.0f}m {:.0f}s\tElapsed Time: {:.0f}m {:.0f}s'.format(
             epoch, time_epoch_elapsed // 60, time_epoch_elapsed % 60,
             time_train_elapsed // 60, time_train_elapsed % 60))

        # Save checkpoint after every epoch if checkpoint_path is given
        if not checkpoint_path == None:
            utils.save_checkpoint(model.state_dict(), optimizer.state_dict(), epoch, checkpoint_path)

    return model # retun trained model