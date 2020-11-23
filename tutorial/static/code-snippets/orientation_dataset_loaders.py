num_workers = 0

# data loader for training
orientation_train_loader = torch.utils.data.DataLoader(
    orientation_train_set, batch_size=4, shuffle=True, num_workers=num_workers
)
# data loader for validation
orientation_valid_loader = torch.utils.data.DataLoader(
    orientation_valid_set, batch_size=4, shuffle=True, num_workers=num_workers
)
# data loader for testing
orientation_test_loader = torch.utils.data.DataLoader(
    orientation_test_set, batch_size=2, shuffle=False, num_workers=num_workers
)

# defining orientation data loaders dictionary
orientation_loaders = {
    'train' : orientation_train_loader,
    'valid' : orientation_valid_loader,
    'test' : orientation_test_loader,
}