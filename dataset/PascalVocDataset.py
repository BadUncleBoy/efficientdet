import os
import torch
import numpy as np

from torch.utils.data import Dataset
import cv2

class PascalVocDataset(Dataset):
    def __init__(self, root_dir, set, img_size, anchor_free_mode=False, transform=None):

        # self.root_dir = root_dir
        self.transform = transform
        self.img_size  = img_size
        self.anchor_free_mode = anchor_free_mode
        anno_path = os.path.join(root_dir, '{0}.txt'.format(set))

        self.lines = self.read_lines(anno_path)
    def parse_line(self, line):
        '''
        Given a line from the training/test txt file, return parsed info.
        line format: line_index, img_path, img_width, img_height, [box_info_1 (5 number)], ...
        return:
            line_idx: int64
            pic_path: string.
            boxes: shape [N, 4], N is the ground truth count, elements in the second
                dimension are [x_min, y_min, x_max, y_max]
            labels: shape [N]. class index.
            img_width: int.
            img_height: int
        '''
        if 'str' not in str(type(line)):
            line = line.decode()
        s = line.strip().split(' ')
        assert len(s) > 8, 'Annotation error! Please check your annotation file. Make sure there is at least one target object in each image.'
        line_idx = int(s[0])
        pic_path = s[1]
        img_width = int(s[2])
        img_height = int(s[3])
        s = s[4:]
        assert len(s) % 5 == 0, 'Annotation error! Please check your annotation file. Maybe partially missing some coordinates?'
        box_cnt = len(s) // 5
        boxes = []
        for i in range(box_cnt):
            label, x_min, y_min, x_max, y_max = int(s[i * 5]), float(s[i * 5 + 1]), float(s[i * 5 + 2]), float(
                s[i * 5 + 3]), float(s[i * 5 + 4])
            boxes.append([x_min, y_min, x_max, y_max, label])
        boxes = np.asarray(boxes, np.float32)
        return pic_path, boxes, img_width, img_height

    def read_lines(self, anno_path):
        lines = []
        with open(anno_path, "r") as f:
            for each in f.readlines():
                lines.append(each)
        return lines


    def __len__(self):
        return len(self.lines)


    def __getitem__(self, idx):
        pic_path, annot, img_width, img_height = self.parse_line(self.lines[idx])
        # print(annot)
        if(self.anchor_free_mode):
            annot = self.gen_anchor_free_annotation(annot, img_height, img_width)
            
        img = self.load_image(pic_path)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def gen_anchor_free_annotation(self, annot, height, width):

        y_true = []
        if height > width:
            scale = self.img_size / height
        else:
            scale = self.img_size / width
        
        for stride in [8, 16, 32, 64, 128]:
            # 4:bbox; 1:is grid empty; 1:empty or not; 20:one-hot code for class
            y_true_i = np.zeros((self.img_size//stride, self.img_size//stride, 4+1+20))
            y_true_i[:, :, 4] = -1
            y_true_i[:, :, :4] = 10
            
            for numth in range(annot.shape[0]):

                box_down = annot[numth, :-1] * scale / stride
                x_min = int(box_down[0]) if box_down[0] < (int(box_down[0]) + 0.5) else int(box_down[0]) + 1
                x_max = int(box_down[2]) if box_down[2] > (int(box_down[2]) + 0.5) else int(box_down[2]) - 1
                y_min = int(box_down[1]) if box_down[1] < (int(box_down[1]) + 0.5) else int(box_down[1]) + 1
                y_max = int(box_down[3]) if box_down[3] > (int(box_down[3]) + 0.5) else int(box_down[3]) - 1
                
                # special contition if the object falls in the rightest gird's right part area
                if x_min == (self.img_size // stride) or y_min == (self.img_size // stride):
                    continue 
                
                # make sure at least one grid contains this object
                x_max = x_min if x_min == (x_max + 1) else x_max
                y_max = y_min if y_min == (y_max + 1) else y_max

                r_distance = annot[numth, 2] / self.img_size
                for axis_1 in range(x_min, x_max+1):
                    for axis_0 in range(y_min, y_max+1):
                        exist_r_distance = y_true_i[axis_0, axis_1, 2]
                        if (r_distance < exist_r_distance):
                            # we need to resize the gt bbox in the Resizer class
                            y_true_i[axis_0, axis_1, 0] = annot[numth, 0] / self.img_size
                            y_true_i[axis_0, axis_1, 1] = annot[numth, 1] / self.img_size
                            y_true_i[axis_0, axis_1, 2] = annot[numth, 2] / self.img_size
                            y_true_i[axis_0, axis_1, 3] = annot[numth, 3] / self.img_size
                            y_true_i[axis_0, axis_1, 4] = 1.
                            y_true_i[axis_0, axis_1, 5:] = 0.
                            y_true_i[axis_0, axis_1, 5+int(annot[numth,4])] = 1.
        
            y_true.append(y_true_i.reshape(-1, 4+1+20))
        anno = np.concatenate(y_true, axis=0)
        
        return anno



    def load_image(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img.astype(np.float32) / 255.
