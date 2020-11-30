segmented_image = utils.segment_image(image, mask)
aligned_student_id = utils.align_student_id(image, mask)

fig = plt.figure(figsize=(20, 15))
ax = fig.add_subplot(1, 2, 1, xticks=[], yticks=[])
ax.set_title('Segmented Image', fontdict={'fontsize': 22})
plt.imshow(segmented_image)

ax1 = fig.add_subplot(1, 2, 2, xticks=[], yticks=[])
ax1.set_title('Aligned Student-Id', fontdict={'fontsize': 22})
plt.imshow(aligned_student_id)