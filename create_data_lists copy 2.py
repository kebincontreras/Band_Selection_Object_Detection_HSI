import os
import xml.etree.ElementTree as ET
import json
import tifffile as tiff

# Label map
voc_labels = ('leaf_screen_hy','toyblock_screen_hy','photo_screen_hy','pen_screen_hy','leaf_real_hy','toyblock_real_hy','photo_real_hy','pen_real_hy')
label_map = {k: v + 1 for v, k in enumerate(voc_labels)}
label_map['background'] = 0

def parse_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    boxes = []
    labels = []
    difficulties = []
    for object in root.iter('object'):
        difficult = int(object.find('difficult').text == '1')
        label = object.find('name').text.lower().strip()
        if label not in label_map:
            continue
        bbox = object.find('bndbox')
        xmin = int(bbox.find('xmin').text) - 1
        ymin = int(bbox.find('ymin').text) - 1
        xmax = int(bbox.find('xmax').text) - 1
        ymax = int(bbox.find('ymax').text) - 1
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label_map[label])
        difficulties.append(difficult)
    return {'boxes': boxes, 'labels': labels, 'difficulties': difficulties}

def filter_images_and_save_json(voc_path, output_folder, desired_dims):
    voc_path = os.path.abspath(voc_path)
    image_sets = ['trainval', 'test']
    for image_set in image_sets:
        images = []
        objects = []
        with open(os.path.join(voc_path, f'ImageSets/Main/{image_set}.txt')) as f:
            ids = f.read().splitlines()
        for id in ids:
            image_path = os.path.join(voc_path, 'JPEGImages', id + '.tiff')
            if os.path.exists(image_path):
                img = tiff.imread(image_path)
                if img.shape == desired_dims:
                    objects_data = parse_annotation(os.path.join(voc_path, 'Annotations', id + '.xml'))
                    if len(objects_data['boxes']) > 0:
                        images.append(image_path)
                        objects.append(objects_data)
        # Save to JSON files with '_a' suffix
        with open(os.path.join(output_folder, f'{image_set}_images.json'), 'w') as j:
            json.dump(images, j)
        with open(os.path.join(output_folder, f'{image_set}_objects.json'), 'w') as j:
            json.dump(objects, j)
        print(f'\n{len(images)} {image_set} images of the desired dimension have been processed and saved.')

if __name__ == '__main__':
    voc07_path = r"C:\Users\USUARIO\Documents\GitHub\Repositorio_otras_personas\HSI-Object-Detection-NPU\data\VOC2007"
    output_folder = './'
    desired_dims = (96, 467, 336)
    filter_images_and_save_json(voc07_path, output_folder, desired_dims)
