import coordinates

# get size of aligned image
image_width = aligned_image.width
image_height = aligned_image.height

im_width = field_coordinates['image_width']
im_height = field_coordinates['image_height']

image_array = np.array(aligned_image.resize((im_width, im_height)))

fig = plt.figure(figsize=(20, 40))
for counter, field_id in enumerate(['firstname', 'lastname', 'photo', 'id']):
    cords = field_coordinates[field_id]
    x1, y1, x2, y2 = cords['left'], cords['top'], cords['width']+cords['left'], \
                        cords['height']+cords['top']
    crop_cords = x1, y1, x2, y2
    cropped_image = utils.crop_image(image_array, crop_cords)
    ax = fig.add_subplot(1, 4, counter+1, xticks=[], yticks=[])
    plt.imshow(cropped_image)
    ax.set_title('{}'.format(field_id))
