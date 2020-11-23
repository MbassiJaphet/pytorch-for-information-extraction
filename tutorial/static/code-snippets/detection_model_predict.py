# pick one image from the test set
id = 7
image = Image.open(detection_test_set.image_urls[id])
print('Image size: ', image.size)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
plt.imshow(image)
ax.set_title('Test Image')

# put the model in evaluation mode
detection_model.eval()
with torch.no_grad():
    # forward pass the test image to get detection predictions
    detection_predictions = detection_model(imgToTensor(image).unsqueeze(0).to(device))