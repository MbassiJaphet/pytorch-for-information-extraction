import modules.detection.scripts.transforms as detection_transforms

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(detection_transforms.ToTensor())
    ### feel free to add additional transforms here below

    return detection_transforms.Compose(transforms)