import modules.detection.scripts.utils as script_utils

# data loader for training
detection_train_loader = torch.utils.data.DataLoader(
    detection_train_set, batch_size=6, shuffle=True, num_workers=4,
    collate_fn=script_utils.collate_fn)

# data loader for validation
detection_valid_loader = torch.utils.data.DataLoader(
    detection_valid_set, batch_size=2, shuffle=False, num_workers=2,
    collate_fn=script_utils.collate_fn)

# data loader for testing
detection_test_loader = torch.utils.data.DataLoader(
    detection_test_set, batch_size=2, shuffle=False, num_workers=0,
    collate_fn=script_utils.collate_fn)

# defining orientation data loaders dictionary
detection_loaders = {
    'train' : detection_train_loader,
    'valid' : detection_valid_loader,
    'test' : detection_test_loader,
}