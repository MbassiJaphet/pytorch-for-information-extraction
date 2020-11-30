# Select loss criterion
orientation_criterion = torch.nn.CrossEntropyLoss()
# Select optimizer
orientation_optimizer = torch.optim.SGD(orientation_model.parameters(), lr=0.001)
# Regularization. Decay LR by a factor of 0.9 every 15 epochs
orientation_lr_scheduler = torch.optim.lr_scheduler.StepLR(orientation_optimizer, step_size=15, gamma=0.9)

# load optimizer state dictionary from checkpoint if available
if orientation_optimizer_state_dict is None:  print('No checkpoint loaded ! Optimizer not loaded from checkpoint...')
else:
    orientation_optimizer.load_state_dict(orientation_optimizer_state_dict)
    print('Loaded optimizer from checkpoint...')