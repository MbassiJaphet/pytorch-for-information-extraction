# activate gradients calculation for unfreezed parameters
detection_params = [p for p in detection_model.parameters() if p.requires_grad]
# initialize training optimizer with learning rate, momentum and weight decay
detection_optimizer = torch.optim.SGD(detection_params, lr=0.005, momentum=0.9, weight_decay=0.0005)
# define learning rate schelduler to gradually decay learning rate
detection_lr_scheduler = torch.optim.lr_scheduler.StepLR(detection_optimizer, step_size=10, gamma=0.95)

# load optimizer state dictionary from checkpoint if available
if detection_optimizer_state_dict is None:  print('No checkpoint loaded ! Optimizer not loaded from checkpoint...')
else:
    detection_optimizer.load_state_dict(detection_optimizer_state_dict)
    print('Loaded optizer from checkpoint...')