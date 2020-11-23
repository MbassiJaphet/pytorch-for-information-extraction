from torchvision import datasets

# define urls of our datasets folders
orientation_data_path = os.path.join('datasets', 'orientation')
orientation_train_images_path = os.path.join(orientation_data_path, 'train')
orientation_valid_images_path = os.path.join(orientation_data_path, 'valid')
orientation_test_images_path = os.path.join(orientation_data_path, 'test')

# initialize training dataset for orientation module
orientation_train_set = datasets.ImageFolder(orientation_train_images_path, orientation_transform_train)
# initialize validation dataset for orientation module
orientation_valid_set = datasets.ImageFolder(orientation_valid_images_path, orientation_transform_valid)
# initialize testing dataset for orientation module
orientation_test_set = datasets.ImageFolder(orientation_test_images_path, orientation_transform_test)
# retrieve orientation classes
orientation_classes = orientation_train_set.classes
num_orientation_classes = len(orientation_classes)