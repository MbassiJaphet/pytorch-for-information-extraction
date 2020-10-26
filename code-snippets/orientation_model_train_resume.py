# resume training of orientation model up to 30 epochs
orientation_model = train_orientation_model(orientation_model, orientation_checkpoint, 30,
                        orientation_optimizer, orientation_criterion, use_cuda = torch.cuda.is_available(),
                        checkpoint_path = orientation_checkpoint_path, loaders = orientation_loaders)