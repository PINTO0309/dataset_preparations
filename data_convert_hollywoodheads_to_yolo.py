"""
This script converts the data from the Hollywoods dataset
in the format that is expected for the training by the train.py script.
"""
from __future__ import division

import xml.etree.ElementTree as ET
import os
import math
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

PHASES = ['train', 'val', 'test']

def resize_and_pad(img, size, pad_color=0):
    h, w = img.shape[:2]
    sh, sw = size
    if h > sh or w > sw:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_CUBIC
    aspect = w/h
    if aspect > 1:
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = \
            np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1:
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = \
            np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else:
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0
    if len(img.shape) == 3 and not isinstance(pad_color, (list, tuple, np.ndarray)):
        pad_color = [pad_color]*3
    scaled_img = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=interp
    )
    scaled_img = cv2.copyMakeBorder(
        scaled_img,
        pad_top,
        pad_bot,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_color
    )
    return scaled_img, pad_left, pad_right, pad_top, pad_bot


def convert_txt(
    root_path,
    src_path,
    dst_path,
    ann_path,
    show,
    resized_images_size_h,
    resized_images_size_w,
    yolo_path,
    yolo_root_path,
    yolo_label
):

    os.makedirs(f'{yolo_path}', exist_ok=True)

    with open(dst_path, "w") as wfp:
        with open(src_path, 'r') as rfp:
            for line in tqdm(rfp.readlines()):
                line, _ = line.split("\n")
                ann_file_name = line + ".xml"
                insert_line = ""
                insert_line = insert_line + line + ".jpg"
                img_name = line + ".jpeg"
                img_path = os.path.join(f'{root_path}/JPEGImages', img_name)
                f = Image.open(img_path)
                f.convert('RGB')
                original_image_size_w, original_image_size_h = f.size

                f = np.array(f, dtype=np.uint8)
                try:
                    img, pad_left, pad_right, pad_top, pad_bot = \
                        resize_and_pad(f, (resized_images_size_h, resized_images_size_w))
                except:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                insert_line = "\"" + insert_line + "\""
                ann = ET.parse(os.path.join(ann_path,ann_file_name))
                bboxs = []
                for obj in ann.findall('object'):
                    bbox_ann = obj.find('bndbox')
                    if bbox_ann is None:
                        continue
                    bboxs.append(
                        [float(bbox_ann.find(tag).text) - 1 for tag in ('xmin', 'ymin', 'xmax', 'ymax')]
                    )
                for i in range(len(bboxs)):
                    bbox = bboxs[i]

                    image_size_final_w = resized_images_size_w - (pad_left + pad_right)
                    image_size_final_h = resized_images_size_h - (pad_top + pad_bot)

                    bbox[0] = int(bbox[0]*(image_size_final_w/original_image_size_w)+pad_left)
                    bbox[1] = int(bbox[1]*(image_size_final_h/original_image_size_h)+pad_top)
                    bbox[2] = int(bbox[2]*(image_size_final_w/original_image_size_w)+pad_left)
                    bbox[3] = int(bbox[3]*(image_size_final_h/original_image_size_h)+pad_top)

                    bbox[0] = 0 if bbox[0] < 0 else resized_images_size_w if bbox[0] > resized_images_size_w else bbox[0]
                    bbox[1] = 0 if bbox[1] < 0 else resized_images_size_h if bbox[1] > resized_images_size_h else bbox[1]
                    bbox[2] = 0 if bbox[2] < 0 else resized_images_size_w if bbox[2] > resized_images_size_w else bbox[2]
                    bbox[3] = 0 if bbox[3] < 0 else resized_images_size_h if bbox[3] > resized_images_size_h else bbox[3]

                if show:
                    view_img = img.copy()
                    for box in bboxs:
                        cv2.rectangle(
                            view_img,
                            pt1=(box[0], box[1]),
                            pt2=(box[2], box[3]),
                            color=(0, 0, 255),
                            thickness=1,
                        )

                    cv2.imshow('image', view_img)
                    if cv2.waitKey(0)&0xFF == ord('q'):
                        break

                bboxs_string = ""
                nbboxs = len(bboxs)
                index = 1
                if nbboxs == 0:
                    pass

                else:
                    # Image
                    cv2.imwrite(
                        os.path.join(f'{yolo_path}', line+".jpg"),
                        img
                    )
                    # IDL
                    for bbox in bboxs:
                        bbox_string = ""
                        if index <= nbboxs - 1:
                            bbox_string = bbox_string + ', '.join(str(math.floor(e)) for e in bbox)
                            bbox_string = '(' + bbox_string + '), '
                            bboxs_string = bboxs_string + bbox_string
                            index += 1
                        else:
                            bbox_string = bbox_string + ', '.join(str(math.floor(e)) for e in bbox)
                            bbox_string = '(' + bbox_string + '); '
                            bboxs_string = bboxs_string + bbox_string
                            index += 1
                    insert_line = insert_line + ': '
                    insert_line = insert_line + bboxs_string + '\n'
                    wfp.write(insert_line)

                    # YOLO
                    insert_line = insert_line.replace(":",";")
                    img_dir = insert_line.split(";")[0]
                    img_boxs = insert_line.split(";")[1]
                    img_dir = img_dir.replace('"',"")
                    img_name = img_dir
                    txt_name = img_name.split(".")[0]
                    img_extension = img_name.split(".")[1]
                    img_boxs = img_boxs.replace(",","")
                    img_boxs = img_boxs.replace("(","")
                    img_boxs = img_boxs.split(")")
                    if(img_extension == 'jpg'):
                        for n in range(len(img_boxs)-1):
                            box = img_boxs[n]
                            box = box.split(" ")
                            with open(f'{yolo_path}/{txt_name}.txt','a') as f:
                                f.write(' '.join(
                                    [
                                        yolo_label,
                                        str((float(box[1]) + float(box[3]))/(2*resized_images_size_w)),
                                        str((float(box[2]) + float(box[4]))/(2*resized_images_size_h)),
                                        str((float(box[3]) - float(box[1]))/resized_images_size_w),
                                        str((float(box[4]) - float(box[2]))/resized_images_size_h)
                                    ]
                                )+'\n')
                        with open(f'{yolo_path}/{os.path.basename(src_path)}','a') as f:
                            f.write(f'{yolo_root_path}/{img_name}\n')

    if show:
        cv2.destroyAllWindows()


def convert_hollywood(
    root_path,
    show,
    resized_images_size_h,
    resized_images_size_w,
    yolo_path,
    yolo_root_path,
    yolo_label
):
    splits_folder = os.path.join(root_path, 'Splits')
    ann_folder = os.path.join(root_path, 'Annotations')
    for phase in PHASES:
        data_list_path = ''
        data_path = ''
        if phase == 'train':
            print("Phase train ongoing...")
            data_list_path = os.path.join(splits_folder, 'train.txt')
            data_path = os.path.join(root_path, 'hollywood_train.idl')

        # elif phase == 'val':
        #     print("Phase val ongoing...")
        #     data_list_path = os.path.join(splits_folder, 'val.txt')
        #     data_path = os.path.join(root_path, 'hollywood_val.idl')

        else:
            print("Phase test ongoing...")
            data_list_path = os.path.join(splits_folder, 'test.txt')
            data_path = os.path.join(root_path, 'hollywood_test.idl')

        convert_txt(
            root_path,
            data_list_path,
            data_path,
            ann_folder,
            show,
            resized_images_size_h,
            resized_images_size_w,
            yolo_path,
            yolo_root_path,
            yolo_label
        )


if __name__ == "__main__":
    root_path = 'HollywoodHeads'
    show = False
    resized_images_size_h = 480
    resized_images_size_w = 640

    yolo_path = f'{root_path}/crowdhuman-{resized_images_size_w}x{resized_images_size_h}'
    yolo_root_path = f'data/crowdhuman-{resized_images_size_w}x{resized_images_size_h}'
    yolo_label = '0' # 0:head, 1:person

    convert_hollywood(
        root_path,
        show,
        resized_images_size_h,
        resized_images_size_w,
        yolo_path,
        yolo_root_path,
        yolo_label
    )