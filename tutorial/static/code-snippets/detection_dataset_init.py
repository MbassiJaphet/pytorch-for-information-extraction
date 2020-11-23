detection_data_path = os.path.join('datasets', 'detection')
# training dataset
detection_train_set = DetectionDataset(detection_data_path, mode='train', transforms=get_transform(True))
# validation dataset
detection_valid_set = DetectionDataset(detection_data_path, mode='valid', transforms=get_transform(True))
#testing dataset
detection_test_set = DetectionDataset(detection_data_path, mode='test', transforms=get_transform(False))

detection_classes = detection_train_set.classes
num_detection_classes  = len(detection_classes) 