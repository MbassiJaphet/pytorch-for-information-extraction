class DetectionDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, mode=None, classes=list(), transforms=None):
        self.mode = mode
        self.data_path = data_path
        self.transforms = transforms
        # everything but our dataset classes ibelong to class 'BACKGROUND'
        self.classes = ['BACKGROUND']
        # loading our dataset classes names
        _classes_names = utils.load_json(os.path.join(data_path, 'classes.json'))['classes']
        # implicitly attributing index '0' to BACKGROUND class
        self.classes.extend(_classes_names)
        # load all image files
        dataset_file = os.path.join(data_path, str(mode).lower().__add__('.json'))
        if not os.path.exists(dataset_file):
            raise Exception("Invalid Mode: '{}'\n Available modes are: 'TRAIN', 'VALID', 'TEST'.".format(mode))
        data_dict = utils.load_json(dataset_file)
        self.image_urls = dict()
        self.annotation_urls = dict()
        for object_id, item_dict in enumerate(data_dict['data']):
            self.image_urls[object_id] = item_dict['image_url'].replace('\\', '/')
            self.annotation_urls[object_id] = item_dict['annotation_url'].replace('\\', '/')

    def __getitem__(self, idx):
        # load images and annotations
        image_url = self.image_urls[idx]
        annotation_url = self.annotation_urls[idx]
        annotation_dict = utils.load_json(annotation_url)
        image = Image.open(image_url)
        image_height, image_width = image.size
        num_objects = len(annotation_dict['shapes'])
        labels, boxes, polygons = list(), list(), list()
        target = dict()

        for idx, shape in enumerate(annotation_dict['shapes']):
            label = self.classes.index(shape['label'].upper())
            polygon = [(int(x), int(y)) for x, y in shape['points']]
            labels.append(label)
            polygons.append(polygon)

        masks_array = np.zeros((image_width, image_height))
        masks_array = utils.draw_polygons_on_image_array(masks_array, polygons)
        object_ids = np.unique(masks_array)[1:]  # Remove index for background
        mask_arrays = masks_array == object_ids[:, None, None]

        for mask_array in mask_arrays:
            box = utils.compute_box_from_mask_array(mask_array)
            boxes.append(box)

        labels_tensor = torch.from_numpy(np.array(labels))
        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        masks_tensor = torch.as_tensor(mask_arrays, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes_tensor[:, 3] - boxes_tensor[:, 1]) * (boxes_tensor[:, 2] - boxes_tensor[:, 0])

        is_crowd = torch.zeros((num_objects,), dtype=torch.int64)

        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = is_crowd
        target["labels"] = labels_tensor
        target["boxes"] = boxes_tensor
        target["masks"] = masks_tensor

        if self.transforms is not None: image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.image_urls)