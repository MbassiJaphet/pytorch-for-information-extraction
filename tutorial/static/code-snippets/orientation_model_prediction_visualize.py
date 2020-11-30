# define function to correct orientation
def rectify_orientation(image, rotation_angle):
    if rotation_angle == '090': return image.transpose(Image.ROTATE_270)
    elif rotation_angle == '180': return image.transpose(Image.ROTATE_180)
    elif rotation_angle == '270': return image.transpose(Image.ROTATE_90)
    else: return image

image = Image.open('images/aligned_image.jpg')
image_tensor =  orientation_transform_test(image)
orientation_prediction = orientation_model(image_tensor.unsqueeze(0))[0]
print('Raw predictions: ', orientation_prediction)

# retrieve index for maximum score
orientation_index = torch.argmax(orientation_prediction)
# retrieve class of maximun score
orientation_class = orientation_classes[orientation_index]

fig = plt.figure(figsize=(25, 25))
ax = fig.add_subplot(1, 2, 1, xticks=[], yticks=[])
ax.set_title('Raw Outputs : {}\nPredicted Orientation : {}'.format(orientation_prediction, orientation_class))
plt.imshow(image)
ax1 = fig.add_subplot(1, 2, 2, xticks=[], yticks=[])
ax1.set_title('Corrected Orientation')
plt.imshow(rectify_orientation(image, orientation_class))