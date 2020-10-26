import random

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(20, 40))
# visualize transformed data
for counter in range(4):
  image_tensor, label_tensor = orientation_train_set[random.choice(range(0, len(orientation_train_set)))]
  image_tensor = image_tensor * 0.226 + 0.445  # denormalize tensor
  ax = fig.add_subplot(1, 4, counter+1, xticks=[], yticks=[])
  plt.imshow(tensorToPIL(image_tensor))
  ax.set_title('Class Index: {} | Class Name: {}'.format(label_tensor, orientation_classes[label_tensor]))