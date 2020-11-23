def get_orientation_model(num_classes, model_state_dict=None):
    orientation_model = OrientationModel(num_classes)
    # initialize model state dictionary if specified
    if not model_state_dict == None: orientation_model.load_state_dict(model_state_dict)

    return orientation_model