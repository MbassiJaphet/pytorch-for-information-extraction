# start training the orientation model for 20 epochs
orientation_model = train_orientation_model(orientation_model, orientation_checkpoint, 20,
                        orientation_optimizer, orientation_criterion, use_cuda = torch.cuda.is_available(),
                        checkpoint_path = orientation_checkpoint_path, loaders = orientation_loaders)