id = 0
obj_id = None
obj_label = ''
fig = plt.figure(figsize=(15, 15))

image_tensor, targets = detection_train_set[id]

boxes = targets['boxes'] # retrieve bounding boxes
image = utils.tensorToPIL(image_tensor)
image_array = np.array(image)

for box in boxes :
  cv2.rectangle(image_array, (box[0],box[1]), (box[2],box[3]), (255,0,0), 2) # draw bounding boxes

ax = fig.add_subplot(1, 2, 1, xticks=[], yticks=[])
plt.imshow(Image.fromarray(image_array))
ax.set_title('Image')

if obj_id is not None:
    mask_tensor = targets['masks'][obj_id] # retrieve bounding masks
    obj_label_idx = targets['labels'][[obj_id]].item() # retrieve bounding labels
    obj_label = ': ' + detection_classes[obj_label_idx]
else :
    mask_tensor = torch.zeros_like(image_tensor)
    for _mask_tensor in targets['masks'] : mask_tensor += _mask_tensor # paste mask for every object

ax1 = fig.add_subplot(1, 2, 2, xticks=[], yticks=[])
ax1.set_title('Segentation Mask' + obj_label)
mask = utils.tensorToPIL(mask_tensor)
plt.imshow(mask)