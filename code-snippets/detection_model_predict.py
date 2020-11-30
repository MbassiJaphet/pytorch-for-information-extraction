# pick one image from the test set
id = 7
image = Image.open(detection_test_set.image_urls[id])
image_tensor = imgToTensor(image)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
ax.set_title('Test Image')
plt.imshow(image)

# put the model in evaluation mode
detection_model.eval()
with torch.no_grad():
    # forward pass the test image to get detection predictions
    detection_predictions = detection_model(image_tensor.unsqueeze(0).to(device))