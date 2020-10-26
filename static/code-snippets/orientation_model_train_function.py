def train_orientation_model(model, checkpoint, num_epochs, optimizer, criterion,
            use_cuda=True, checkpoint_path=None, loaders=None, scheduler=None):
    # Specify hardware type for training
    device = torch.device('cuda') if use_cuda else torch.device('cpu')

    best_model_state_dict = None
    best_optimizer_state_dict = None

    if checkpoint == None:
        start_epoch = 1
        best_loss = np.inf
    else:
        print('Resuming training from checkpoint...')
        best_loss = checkpoint['best_loss']
        model_state_dict = checkpoint['model_state_dict']
        optimizer_state_dict = checkpoint['optimizer_state_dict']
        best_model_state_dict = model_state_dict
        best_optimizer_state_dict = optimizer_state_dict
        start_epoch = checkpoint['epoch'] + 1
    # move model to hardware
    model.to(device)

    # initialize tracker for minimum validation loss
    valid_loss_min = best_loss

    time_train = time.time()
    for epoch in range(start_epoch, num_epochs + 1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        time_epoch = time.time()

        ### Defining training block
        # set model to training mode
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move data to hardware
            data, target = data.to(device), target.to(device)
            ## find the training loss and update the model parameters accordingly
            # reset gradients
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            # record the average training loss
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

        if scheduler is not None: scheduler.step()

        ### Defining validation block
        # set model for evaluation mode
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move data to hardware
            data, target = data.to(device), target.to(device)
            ## find the validation
            output = model(data)
            loss = criterion(output, target)
            # update the average validation loss
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

        # print training and validation statistics
        time_epoch_elapsed = time.time() - time_epoch
        time_train_elapsed = time.time() - time_train
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tEpoch Time: {:.0f}m {:.0f}s \tElapsed Time: {:.0f}m {:.0f}s'
                .format(epoch, train_loss, valid_loss, time_epoch_elapsed // 60, time_epoch_elapsed % 60,
                time_train_elapsed // 60, time_train_elapsed % 60))

        ## update checkpoint if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving best checkpoint ...'
                    .format(valid_loss_min, valid_loss))
            ## update checkpoint if checkpoint path is given
            if checkpoint_path is not None:
                best_training_epoch = epoch
                best_model_state_dict = model.state_dict()
                best_optimizer_state_dict = optimizer.state_dict()
                utils.save_checkpoint(best_model_state_dict, best_optimizer_state_dict,
                                      best_training_epoch, checkpoint_path, loss= valid_loss)
            valid_loss_min = valid_loss

    ## update checkpoint if checkpoint path is given
    if checkpoint_path is not None:
        if best_model_state_dict == None and best_optimizer_state_dict:
            print('No validation loss decrease ! Saving checkpoint ...')
            utils.save_checkpoint(model.state_dict(), optimizer.state_dict(), num_epochs,
                                  checkpoint_path, loss= valid_loss_min)
        else:
            print('Saving checkpoint ...')
            utils.save_checkpoint(best_model_state_dict, optimizer.state_dict(), num_epochs,
                                  checkpoint_path, loss= valid_loss_min)
    # return trained model
    print('Training Complete ...')
    return model