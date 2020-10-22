import os
import io
import csv
import json
import string
import random

import cv2
import glob
import torch
import torchvision
import numpy as np

from shutil import copyfile
from PIL import Image, ImageDraw, ImageFilter


CPU_DEVICE = torch.device('cpu')
CUDA_DEVICE = torch.device('cuda')

imgToTensor = torchvision.transforms.ToTensor()
tensorToPIL = torchvision.transforms.ToPILImage()

def is_cuda_device(device):
    return True if device == CUDA_DEVICE else False


def crop_image(image_array, crop_cords):
    start_col, start_row, end_col, end_row = crop_cords
    return image_array[start_row:end_row, start_col:end_col]


def checkpoint_summary(checkpoint):
    print('Epochs: {} | Best Loss: {}'.format(checkpoint['epoch'], checkpoint['best_loss']))


def get_checkpoint(checkpoint_path):
    checkpoint = checkpoint_path if os.path.exists(checkpoint_path) else None
    return checkpoint


def filter_array(elements, *filters):
    filter_elements = list()
    for filter in filters:
        filter_elements.extend(filter)
    return [element for element in elements if not filter_elements.__contains__(element)]


def clear_dir(mydir, ext):
    files_list = [f for f in os.listdir(mydir) if f.endswith(ext)]
    for f in files_list:
        os.remove(os.path.join(mydir, f))


def mkdir(my_dir):
    if os.path.exists(my_dir):
        print("'{}' folder already exists. Skipping creation...".format(os.path.basename(my_dir)))
    else:
        try:
            os.mkdir(my_dir)
        except OSError:
            print("Failed to directory '{}'. An unexpected error occurred.".format(os.path.basename(my_dir)))
        else:
            print("Directory '{}' successfully created".format(os.path.basename(my_dir)))


def is_json(json_string):
    try:
        json_object = json.loads(json_string)
    except ValueError as e:
        return False
    return json_object


def load_json(json_url):
    with open(json_url) as json_file:
        my_dict = json.load(json_file)
        json_file.close()
    return my_dict


def dump_json(object, target_file):
    json_file = open(target_file, "w+")
    json.dump(object, json_file, indent=4)
    json_file.close()


def save_checkpoint(model_state_dict, optimizer_state_dict, epoch, checkpoint_path, loss= None):
    state = {
        'epoch': epoch,
        'best_loss': loss,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict
    }
    torch.save(state, checkpoint_path)


def convert_polygon_to_mask(size, polygons):
    img = Image.new('L', (size[0], size[1]), 0)
    for ids, polygon in enumerate(polygons):
        ImageDraw.Draw(img).polygon(polygon, outline=ids + 1, fill=ids + 1)
    mask = np.array(img)

    return mask


def draw_polygons_on_image(image, polygons):
    for idx, polygon in enumerate(polygons):
        ImageDraw.Draw(image).polygon(polygon, outline=idx + 1, fill=idx + 1)  # Skipping background '0'

    return image


def draw_polygons_on_image_array(image_array, polygons):
    image = Image.fromarray(image_array)
    image = draw_polygons_on_image(image, polygons)
    image_array = np.array(image)

    return image_array


def create_detection_data(data_path, label_folders, test_set_ratio=0.2, valid_set_ratio=0.05, force_creation=False):
    classes = set()
    train_image_urls = list()
    valid_image_urls = list()
    test_image_urls = list()
    train_set_url = os.path.join(data_path, 'train.json')
    valid_set_url = os.path.join(data_path, 'valid.json')
    test_set_url = os.path.join(data_path, 'test.json')
    classes_url = os.path.join(data_path, 'classes.json')

    if os.path.exists(train_set_url) and os.path.exists(valid_set_url) and os.path.exists(
            test_set_url) and not force_creation:
        print('Existing dataset detected ! Skipping creation...')
        train_image_urls = load_json(train_set_url)['data']
        valid_image_urls = load_json(valid_set_url)['data']
        test_image_urls = load_json(test_set_url)['data']
        print('Dataset loaded successfully !')
    else:
        if not force_creation: print('No existing dataset detected ! Creating dataset...')
        for label_folder in label_folders:
            image_urls = list()
            images_folder = os.path.join(data_path, label_folder)
            annotation_urls = glob.glob(os.path.join(images_folder, '*.json'))
            for annotation_url in annotation_urls:
                image_url = os.path.splitext(annotation_url)[0].__add__('.jpg')
                if not os.path.exists(image_url): raise Exception(
                    "Image file '{}' not found for annotation file '{}'.".format(image_url, annotation_url))
                image_urls.append(image_url)

                annotation_dict = load_json(annotation_url)
                for shape in annotation_dict['shapes']: classes.update([shape['label'].upper()])

            random.shuffle(image_urls)
            num_images = len(image_urls)
            num_valid_images = int(num_images * valid_set_ratio)
            num_test_images = int(num_images * test_set_ratio)

            new_valid_image_urls = image_urls[:num_valid_images]
            random.shuffle(new_valid_image_urls)
            new_test_image_urls = filter_array(image_urls, new_valid_image_urls)[:num_test_images]
            random.shuffle(new_test_image_urls)
            new_train_image_urls = filter_array(image_urls, new_valid_image_urls, new_test_image_urls)
            random.shuffle(new_train_image_urls)

            train_image_urls.extend(new_train_image_urls)
            valid_image_urls.extend(new_valid_image_urls)
            test_image_urls.extend(new_test_image_urls)

        dump_dataset_to_json_file(train_image_urls, train_set_url)
        dump_dataset_to_json_file(valid_image_urls, valid_set_url)
        dump_dataset_to_json_file(test_image_urls, test_set_url)
        dump_json({'num_classes': len(classes), 'classes': sorted(list(classes))}, classes_url)

        print('Dataset created successfully !')
    print('Total Images Detected: {}'.format(len(train_image_urls) + len(valid_image_urls) + len(test_image_urls)))
    print('Train Images: {} | Validation Images: {} | Test Images: {}'.format(len(train_image_urls),
                                                                              len(valid_image_urls),
                                                                              len(test_image_urls)))

    return train_image_urls, valid_image_urls, test_image_urls


def create_orientation_data(data_path, label_folders, test_set_ratio=0.2, valid_set_ratio=0.05, face='', reset=False):
    labels = ['360', '090', '180', '270']

    for label in labels:
        test_folder = os.path.join(data_path, 'test', label)
        train_folder = os.path.join(data_path, 'train', label)
        valid_folder = os.path.join(data_path, 'valid', label)

        if reset:
            for folder in [train_folder, valid_folder, test_folder]:
                clear_dir(folder, '.png')

    for label_folder in label_folders:
        print('Browsing Labels in Folder: {}'.format(label_folder))

        # ''''
        for label in labels:
            test_folder = os.path.join(data_path, 'test', label)
            train_folder = os.path.join(data_path, 'train', label)
            valid_folder = os.path.join(data_path, 'valid', label)

            print(' ->Loading Images For The Label: {}'.format(label.upper()))

            image_urls = np.array(glob.glob(os.path.join(data_path, label_folder, face, label, '*.png')))
            random.shuffle(image_urls)

            num_images = len(image_urls)

            num_valid_images = int(num_images * valid_set_ratio)
            num_test_images = int(num_images * test_set_ratio)

            valid_image_urls = image_urls[:num_valid_images]
            test_image_urls = filter_array(image_urls, valid_image_urls)[:num_test_images]
            train_image_urls = filter_array(image_urls, valid_image_urls, test_image_urls)

            print('   -->Loading Validation Images !')
            for id, image_url in enumerate(valid_image_urls):
                image_name = os.path.basename(image_url)
                target_file = os.path.join(valid_folder, image_name)
                print('      -->{} - {}'.format(id + 1, target_file))
                copyfile(image_url, target_file)

            print('   -->Loading Testing Images !')
            for id, image_url in enumerate(test_image_urls):
                image_name = os.path.basename(image_url)
                target_file = os.path.join(test_folder, image_name)
                print('      -->{} - {}'.format(id + 1, target_file))
                copyfile(image_url, target_file)

            print('   -->Loading Training Images !')
            for id, image_url in enumerate(train_image_urls):
                image_name = os.path.basename(image_url)
                target_file = os.path.join(train_folder, image_name)
                print('      -->{} - {}'.format(id + 1, target_file))
                copyfile(image_url, target_file)


def compute_box_from_mask(mask):
    mask_array = np.array(mask)
    box = compute_box_from_mask_array(mask_array)

    return box


def compute_box_from_mask_tensor(mask_tensor):
    mask = tensorToPIL(mask_tensor)
    box = compute_box_from_mask(mask)

    return box


def compute_box_from_mask_array(mask_array):
    pos = np.where(mask_array)
    xmin = np.min(pos[1])
    xmax = np.max(pos[1])
    ymin = np.min(pos[0])
    ymax = np.max(pos[0])
    box = xmin, ymin, xmax, ymax

    return box


def compute_box_from_points(points):
    xmin, ymin, xmax, ymax = float("inf"), float("inf"), float("-inf"), float("-inf")
    for x, y in points:
        if x < xmin: xmin = x
        if y < ymin: ymin = y
        # Set max coords
        if x > xmax:
            xmax = x
        elif y > ymax:
            ymax = y
    box = xmin, ymin, xmax, ymax

    return box


def dump_dataset_to_json_file(my_list, target_file):
    my_dict = {
        'data': [
            {
                'image_url': image_url,
                'annotation_url': os.path.splitext(image_url)[0].__add__('.json')
            } for image_url in my_list
        ]
    }
    my_file = open(target_file, "w+")
    json.dump(my_dict, my_file, indent=4)
    my_file.close()


def segment_image(image, mask):
    segmented_image = Image.new('RGB', image.size, 'white')
    segmented_image.paste(image, (0, 0), mask)

    return segmented_image


def segment_image_tensor(image_tensor, mask_tensor):
    image, mask = tensorToPIL(image_tensor), tensorToPIL(mask_tensor)
    segmented_image_tensor = imgToTensor(segment_image(image, mask))

    return segmented_image_tensor


def segment_image_array(image_array, mask_array):
    image, mask = Image.fromarray(image_array), Image.fromarray(mask_array)
    segmented_image_array = np.array(segment_image(image, mask))

    return segmented_image_array


def denoise_mask(mask):
    mask = mask.filter(ImageFilter.BLUR)
    mask_array = np.array(mask)
    denoised_mask_array = denoise_mask_array(mask_array)
    denoised_mask = Image.fromarray(denoised_mask_array)
    return denoised_mask


def denoise_mask_tensor(mask_tensor):
    mask = tensorToPIL(mask_tensor)
    denoised_mask = denoise_mask(mask)
    denoided_mask_tensor = imgToTensor(denoised_mask)
    return denoided_mask_tensor


def denoise_mask_array(mask_array):
    _, denoised_mask_array = cv2.threshold(mask_array, 128, 256, cv2.THRESH_BINARY)
    return denoised_mask_array

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def warp_perspective_image(image_array, rect):
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image_array, M, (maxWidth, maxHeight))

    return cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)


def warp_perpective_image_from_mask(image_array, mask_array):
    nonzeros = np.nonzero(mask_array)
    X, Y = nonzeros[1], nonzeros[0]
    points = np.vstack((X, Y)).T
    rect = order_points(points)
    perspective_image_array = warp_perspective_image(image_array, rect)

    return perspective_image_array


if  __name__ == '__main__':
    # create_detection_data('datasets\\detection', ['student-id'], test_set_ratio=0.1, valid_set_ratio=0.1, force_creation=True)
    create_orientation_data('datasets\\orientation', ['student-id'], test_set_ratio=0.15, valid_set_ratio=0.1, reset=True)
    pass
