# resume training the detection model up to 30 epochs
detection_model = train_detection_model(detection_model, num_epochs= 30, loaders= detection_loaders,
                        checkpoint= detection_checkpoint, checkpoint_path= detection_checkpoint_path,
                        optimizer= detection_optimizer, lr_scheduler= detection_lr_scheduler,
                        print_freq=10, device= device)