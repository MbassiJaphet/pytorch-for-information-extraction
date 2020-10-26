# Select an index for a detected/predicted object 
idx = 0

image_array = np.array(image)
# Retrieve predicted bounding box for detected object
box_tensor = detection_predictions[0]['boxes'][idx]
box = [int(cord) for cord in box_tensor.cpu().numpy()]
# draw predicted bounding box on image
cv2.rectangle(image_array, (box[0],box[1]), (box[2],box[3]), (255,0,0), 2)

# Retrieve predicted mask for detected object
mask_tensor = detection_predictions[0]['masks'][idx]
mask_tensor = utils.denoise_mask_tensor(mask_tensor.cpu())
mask = tensorToPIL(mask_tensor)

# Retrieve predictied label for detected object
label_idx = detection_predictions[0]['labels'][idx].item()
label = detection_classes[label_idx]

# Retrieve predictied score for detected object
score = detection_predictions[0]['scores'][idx].item()

fig = plt.figure(figsize=(20, 15))
ax = fig.add_subplot(1, 2, 1, xticks=[], yticks=[])
plt.imshow(image_array)
ax.set_title('Object Detected: {} | Score: {}'.format(label, score))
ax1 = fig.add_subplot(1, 2, 2, xticks=[], yticks=[])
plt.imshow(mask)
ax1.set_title('Segentation Mask')