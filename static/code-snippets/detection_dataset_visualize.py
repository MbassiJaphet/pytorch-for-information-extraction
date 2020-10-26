## select an image from the detection training dataset
id = 0
obj_id = None
obj_label = ''
fig = plt.figure(figsize=(15, 15))

# retrieve the image tensor and its corresponding target tensor
image_tensor, targets = detection_train_set[id]

# targeted object bounding boxes
boxes = targets['boxes']
image = utils.tensorToPIL(image_tensor)
image_array = np.array(image)

# draw bounding box on image
for box in boxes :
  cv2.rectangle(image_array, (box[0],box[1]), (box[2],box[3]), (255,0,0), 2)

ax = fig.add_subplot(1, 2, 1, xticks=[], yticks=[])
plt.imshow(Image.fromarray(image_array))
ax.set_title('Image')

if obj_id is not None:
    # targeted object mask
    mask_tensor = targets['masks'][obj_id]
    # targeted object label
    obj_label_idx = targets['labels'][[obj_id]].item()
    obj_label = ': ' + detection_classes[obj_label_idx]
else :
    mask_tensor = torch.zeros_like(image_tensor)
    # targeted objects masks
    for _mask_tensor in targets['masks'] : mask_tensor += _mask_tensor

mask = utils.tensorToPIL(mask_tensor)

ax1 = fig.add_subplot(1, 2, 2, xticks=[], yticks=[])
plt.imshow(mask)
ax1.set_title('Segentation Mask' + obj_label)