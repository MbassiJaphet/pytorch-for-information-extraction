segmented_image = utils.segment_image(image, mask)

fig = plt.figure(figsize=(20, 15))
ax = fig.add_subplot(1, 2, 1, xticks=[], yticks=[])
plt.imshow(segmented_image)
ax.set_title('Segmented Image')

# Extract quadreploid of segmented image
X = np.nonzero(mask)[1]
Y = np.nonzero(mask)[0]
points = np.vstack((X, Y)).T
rect = utils.order_points(points)

# align detected student id
image_array = np.array(segmented_image)
aligned_image_array = utils.warp_perspective_image(image_array, rect)
aligned_image_array = cv2.cvtColor(aligned_image_array, cv2.COLOR_BGR2RGB)

ax1 = fig.add_subplot(1, 2, 2, xticks=[], yticks=[])
plt.imshow(aligned_image_array)
ax1.set_title('Aligned Image')