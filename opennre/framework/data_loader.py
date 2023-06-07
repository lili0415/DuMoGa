import torch
import torch.utils.data as data
import os, random, json, logging
import numpy as np
import sklearn.metrics
import json
from PIL import Image

_M_RGB2YUV = [[0.299, 0.587, 0.114], [-0.14713, -0.28886, 0.436], [0.615, -0.51499, -0.10001]]
class SentenceREDataset_t(data.Dataset):
    """
    Sentence-level relation extraction dataset
    """

    def __init__(self, text_path, rel_path, pic_path, rel2id, tokenizer, kwargs):
        """
        Args:
            path: path of the input file
            rel2id: dictionary of relation->id mapping
            tokenizer: function of tokenizing
        """
        super().__init__()
        self.text_path = text_path
        self.pic_path = pic_path
        self.rel_path = rel_path
        self.tokenizer = tokenizer
        self.rel2id = rel2id
        self.kwargs = kwargs
        self.image_root='./benchmark/ours/storage/yqin/contest/OpenPSG/submission/panseg/'


        # Load the text file
        self.panseg=json.load(open("./benchmark/ours/for_pan_seg.json","r"))
        logging.info(
            "Loaded pan_seg information with {} images.".format(len(self.panseg)))
        
        self.data = json.load(open(text_path,"r"))  # 'file_name' 'sentence'
        #self.data = self.data['data']
        # f = open(text_path, encoding='UTF-8')
        # self.data = []
        # f_lines = f.readlines()
        # for i1 in range(len(f_lines)):
        #     line = f_lines[i1].rstrip()
        #     if len(line) > 0:
        #         dic1 = eval(line)
        #         self.data.append(dic1)
        # f.close()
        logging.info(
            "Loaded sentence RE dataset {} with {} lines and {} relations.".format(text_path, len(self.data),
                                                                                   len(self.rel2id)))

        # Load the pic feature file
        self.img_dict = {}
        self.bbox={}
        self.img_feat = 1024
        self.obj_num = 10
        zero_list = list(np.zeros([self.img_feat]))
        zero_box = list(np.zeros([4]))
        self.pic_file_list = os.listdir(self.pic_path)
        for pic_file in self.pic_file_list:
            read = json.load(open(os.path.join(self.pic_path, pic_file),"r")) 
            file_name = read['file_name']
            labels = read['labels']
            self.bbox[pic_file]=read['bboxes']
            roifeatures = read['roifeatures']  # 1024*n
            if len(labels) < self.obj_num:
                for k in range(len(labels), self.obj_num):
                         roifeatures.append(zero_list)
                         self.bbox[pic_file].append(zero_box)
            self.bbox[pic_file]=np.array(self.bbox[pic_file])[:self.obj_num,:]
            self.img_dict[pic_file] = np.array(roifeatures)[:self.obj_num,:]        
        # for pic_file in self.pic_file_list:
        #     with open(os.path.join(self.pic_path, pic_file), 'r', encoding='utf-8') as f:
        #         line_list = f.readlines()
        #         feature_list = line_list[2].strip().split('\t')
        #         class_list = line_list[1].strip().split('\t')
        #         if class_list == ['']:
        #             feature_list = zero_list * self.obj_num
        #         elif len(class_list) < self.obj_num:
        #             for k in range(len(class_list), self.obj_num):
        #                 feature_list.extend(zero_list)
        #         feature_list = [float(feature) for feature in feature_list]
        #         split_feature_list = [feature_list[index:index + self.img_feat] for index in
        #                               range(0, len(feature_list), self.img_feat)]
        #         self.img_dict[pic_file] = split_feature_list[:self.obj_num]
        logging.info("Loaded image feature dataset {} with {} objects.".format(pic_path, len(self.img_dict.keys())))

        # Load the Structural weight
        self.rel_dict = {}
        self.length = 128
        rel_zero_list = list(np.zeros([self.obj_num]))
        self.rel_file_list = os.listdir(self.rel_path)
        print(self.rel_path)
        for rel_file in self.rel_file_list:
            with open(os.path.join(self.rel_path, rel_file), 'r', encoding='utf-8') as f:
                line_list = f.readlines()[1:]
                rel_list = []
                for line in line_list:
                    line = line.strip().split(' ')
                    line = [float(element) for element in line]
                    rel_list.append(line)
                if len(rel_list) < self.length:
                    for i in range(len(rel_list), self.length):
                        rel_list.append(rel_zero_list)
                img_id = os.path.splitext(os.path.split(rel_file)[1])[0] + '.jpg'
                self.rel_dict[rel_file] = rel_list
                #print(rel_file)
        logging.info("Loaded sentence RE dataset aligned weights with {} samples.".format(len(self.rel_dict.keys())))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        file_name = item['file_name']
        
        img_id = os.path.split(file_name)[1]
        sentence = item['sentence']
        target = item['bboxes']
        max_iou=-1
        pos=-1
        symbol=True
        if img_id+'.json' not in self.bbox.keys():
            img_id='000000495476.jpg'
            symbol=False
        bbox=self.bbox[img_id+'.json']
        for i in range(len(self.bbox[img_id+'.json'])):
            own_bbox=self.bbox[img_id+'.json'][i]
            #print("own_box: ",i,own_bbox)
            boxA=target
            boxB=own_bbox
            boxA = [int(x) for x in boxA]
            boxB = [int(x) for x in boxB]
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])

            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

            boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
            boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
            
            iou = interArea / float(boxAArea + boxBArea - interArea)
            if iou>max_iou:
                max_iou=iou
                pos=i
        target_list = np.zeros(self.obj_num)
        target_list[pos]=1
        tokens, att_mask = self.tokenizer(sentence, **self.kwargs) # tokens, att_masks
        # pos1 = seq[-1]
        # pos2 = seq[-2]
        pic = self.img_dict[img_id+'.json']
        rel = self.rel_dict[str(index)]
        # rel = self.padding(item, rel, pos1, pos2)

        np_pic = np.array(pic).astype(np.float32)
        np_rel = np.array(rel).astype(np.float32)

        pic1 = torch.tensor(np_pic)
        rel1 = torch.tensor(np_rel)

        # list_p = list(pic1)
        # list_r = list(rel1)
        # res = [self.rel2id[item['relation']]] + seq + list_p + list_r

        return symbol,bbox,target_list, tokens.squeeze(dim=0), att_mask.squeeze(dim=0), pic1, rel1 # label, seq1, seq2, ...,pic

    def convert_PIL_to_numpy(self,image, format):
        """
        Convert PIL image to numpy array of target format.

        Args:
            image (PIL.Image): a PIL image
            format (str): the format of output image

        Returns:
            (np.ndarray): also see `read_image`
        """
        if format is not None:
            # PIL only supports RGB, so convert to RGB and flip channels over below
            conversion_format = format
            if format in ["BGR", "YUV-BT.601"]:
                conversion_format = "RGB"
            image = image.convert(conversion_format)
        image = np.asarray(image)
        # PIL squeezes out the channel dimension for "L", so make it HWC
        if format == "L":
            image = np.expand_dims(image, -1)

        # handle formats not supported by PIL
        elif format == "BGR":
            # flip channels if needed
            image = image[:, :, ::-1]
        elif format == "YUV-BT.601":
            image = image / 255.0
            image = np.dot(image, np.array(_M_RGB2YUV).T)

        return image

    def rgb2id(self,color):
        if isinstance(color, np.ndarray) and len(color.shape) == 3:
            if color.dtype == np.uint8:
                color = color.astype(np.int32)
            return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
        return int(color[0] + 256 * color[1] + 256 * 256 * color[2])
    
    def bb_intersection_over_union(self,total_bbox):
        boxA=total_bbox[0]
        boxB=total_bbox[1]
        boxA = [int(x) for x in boxA]
        boxB = [int(x) for x in boxB]

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        
        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou

    def padding(self, item, rel, pos1, pos2):
        pos_head = item['h']['pos']
        pos_tail = item['t']['pos']

        pos_min = pos_head
        pos_max = pos_tail
        if pos_head[0] > pos_tail[0]:
            pos_min = pos_tail
            pos_max = pos_head
            rev = True
        else:
            rev = False
        if not rev:
            pos1_len = pos_min[1] - pos_min[0]
            pos2_len = pos_max[1] - pos_max[0]
        else:
            pos2_len = pos_min[1] - pos_min[0]
            pos1_len = pos_max[1] - pos_max[0]
        pad_pos = [0, pos1, pos1 + pos1_len + 1, pos2, pos2_len + pos2 + 1]
        rel_zero_list = list(np.zeros([self.obj_num]))
        for pos in pad_pos:
            rel.insert(pos, rel_zero_list)
        rel = rel[:self.length]
        return rel

    def collate_fn(data):
        data = list(zip(*data))
        labels = data[0]
        seqs = data[1:]
        batch_labels = torch.tensor(labels).long()  # (B)
        batch_seqs = []
        for seq in seqs:
            batch_seqs.append(torch.cat(seq, 0))  # (B, L)
        return [batch_labels] + batch_seqs

    def eval(self, pred_result, use_name=False):
        """
        Args:
            pred_result: a list of predicted label (id)
                Make sure that the `shuffle` param is set to `False` when getting the loader.
            use_name: if True, `pred_result` contains predicted relation names instead of ids
        Return:
            {'acc': xx}
        """
        correct = 0
        total = len(self.data)
        correct_positive = 0
        pred_positive = 0
        gold_positive = 0

        neg = -1
        for name in ['NA', 'na', 'no_relation', 'Other', 'Others', 'none', 'None']:
            if name in self.rel2id:
                if use_name:
                    neg = name
                else:
                    neg = self.rel2id[name]
                break
        for i in range(total):
            if use_name:
                golden = self.data[i]['relation']
            else:
                golden = self.rel2id[self.data[i]['relation']]

            if golden == pred_result[i]:
                correct += 1
                if golden != neg:
                    correct_positive += 1
            if golden != neg:
                gold_positive += 1
            if pred_result[i] != neg:
                pred_positive += 1
        acc = float(correct) / float(total)
        try:
            micro_p = float(correct_positive) / float(pred_positive)
        except:
            micro_p = 0
        try:
            micro_r = float(correct_positive) / float(gold_positive)
        except:
            micro_r = 0
        try:
            micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)
        except:
            micro_f1 = 0

        result = {'acc': acc, 'micro_p': micro_p, 'micro_r': micro_r, 'micro_f1': micro_f1}
        logging.info('Evaluation result: {}.'.format(result))
        return result


class SentenceREDataset_v(data.Dataset):
    """
    Sentence-level relation extraction dataset
    """

    def __init__(self, text_path, rel_path, pic_path, rel2id, tokenizer, kwargs):
        """
        Args:
            path: path of the input file
            rel2id: dictionary of relation->id mapping
            tokenizer: function of tokenizing
        """
        super().__init__()
        self.text_path = text_path
        self.pic_path = pic_path
        self.rel_path = rel_path
        self.tokenizer = tokenizer
        self.rel2id = rel2id
        self.kwargs = kwargs
        self.image_root='./benchmark/ours/storage/yqin/contest/OpenPSG/submission/panseg/'
        self.image_root_val='./benchmark/ours/storage/yqin/contest/OpenPSG/submission/panseg_val/'


        # Load the text file
        self.panseg=json.load(open("./benchmark/ours/for_pan_seg.json","r"))
        logging.info(
            "Loaded pan_seg information with {} images.".format(len(self.panseg)))

        self.panseg_val=json.load(open("./benchmark/ours/for_pan_seg_val.json","r"))
        logging.info(
            "Loaded pan_seg_val information with {} images.".format(len(self.panseg_val)))

        self.psg=json.load(open("./benchmark/ours/psg.json","r"))
        self.psg=self.psg['data']
        logging.info(
            "Loaded psg information with {} images.".format(len(self.psg)))
        
        self.data = json.load(open(text_path,"r"))  # 'file_name' 'sentence'
        #self.data = self.data['data']
        # f = open(text_path, encoding='UTF-8')
        # self.data = []
        # f_lines = f.readlines()
        # for i1 in range(len(f_lines)):
        #     line = f_lines[i1].rstrip()
        #     if len(line) > 0:
        #         dic1 = eval(line)
        #         self.data.append(dic1)
        # f.close()
        logging.info(
            "Loaded sentence RE dataset {} with {} lines and {} relations.".format(text_path, len(self.data),
                                                                                   len(self.rel2id)))

        # Load the pic feature file
        self.img_dict = {}
        self.bbox={}
        self.img_feat = 1024
        self.obj_num = 10
        #zero_list = list(np.zeros([self.img_feat]))
        zero_list = list(np.zeros(self.img_feat))
        zero_box = list(np.zeros(4))
        self.pic_file_list = os.listdir(self.pic_path)
        for pic_file in self.pic_file_list:
            read = json.load(open(os.path.join(self.pic_path, pic_file),"r")) 
            file_name = read['file_name']
            labels = read['labels']
            self.bbox[pic_file]=read['bboxes']
            roifeatures = read['roifeatures']  # 1024*n
            if len(labels) < self.obj_num:
                for k in range(len(labels), self.obj_num):
                         roifeatures.append(zero_list)
                         self.bbox[pic_file].append(zero_box)
            self.bbox[pic_file]=np.array(self.bbox[pic_file])[:self.obj_num,:]
            self.img_dict[pic_file] = np.array(roifeatures)[:self.obj_num,:]        
        # for pic_file in self.pic_file_list:
        #     with open(os.path.join(self.pic_path, pic_file), 'r', encoding='utf-8') as f:
        #         line_list = f.readlines()
        #         feature_list = line_list[2].strip().split('\t')
        #         class_list = line_list[1].strip().split('\t')
        #         if class_list == ['']:
        #             feature_list = zero_list * self.obj_num
        #         elif len(class_list) < self.obj_num:
        #             for k in range(len(class_list), self.obj_num):
        #                 feature_list.extend(zero_list)
        #         feature_list = [float(feature) for feature in feature_list]
        #         split_feature_list = [feature_list[index:index + self.img_feat] for index in
        #                               range(0, len(feature_list), self.img_feat)]
        #         self.img_dict[pic_file] = split_feature_list[:self.obj_num]
        logging.info("Loaded image feature dataset {} with {} objects.".format(pic_path, len(self.img_dict.keys())))

        # Load the Structural weight
        self.rel_dict = {}
        self.length = 128
        rel_zero_list = list(np.zeros([self.obj_num]))
        self.rel_file_list = os.listdir(self.rel_path)
        print(self.rel_path)
        for rel_file in self.rel_file_list:
            with open(os.path.join(self.rel_path, rel_file), 'r', encoding='utf-8') as f:
                line_list = f.readlines()[1:]
                rel_list = []
                for line in line_list:
                    line = line.strip().split(' ')
                    line = [float(element) for element in line]
                    rel_list.append(line)
                if len(rel_list) < self.length:
                    for i in range(len(rel_list), self.length):
                        rel_list.append(rel_zero_list)
                img_id = os.path.splitext(os.path.split(rel_file)[1])[0] + '.jpg'
                self.rel_dict[rel_file] = rel_list
                #print(rel_file)
        logging.info("Loaded sentence RE dataset aligned weights with {} samples.".format(len(self.rel_dict.keys())))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        file_name = item['file_name']
        target = item['bboxes']
        img_id = os.path.split(file_name)[1]
        sentence = item['sentence']
        bbox=self.bbox[img_id+'.json']

        max_iou=-1
        index_pos=-1
        for i in range(len(self.bbox[img_id+'.json'])):
            own_bbox=self.bbox[img_id+'.json'][i]
            #print("own_box: ",i,own_bbox)
            boxA=target
            boxB=own_bbox
            boxA = [int(x) for x in boxA]
            boxB = [int(x) for x in boxB]
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])

            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

            boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
            boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
            
            iou = interArea / float(boxAArea + boxBArea - interArea)
            if iou>max_iou:
                max_iou=iou
                index_pos=i

        sym=0
        ori_masks = []
        for num,seg in enumerate(self.psg):
            #print(seg['file_name'])
            if file_name==seg['file_name']:
                sym=1
                seg_img=seg['pan_seg_file_name']
                #print(seg_img)
                seg_img=Image.open(self.image_root + seg_img)

                seg_map = self.convert_PIL_to_numpy(seg_img, "RGB")
                seg_map=self.rgb2id(seg_map)
                for i, s in enumerate(seg["segments_info"]):
                    ori_masks.append((seg_map == s["id"]).tolist())

                total_bbox=seg['annotations']
                max_iou=-1
                pos=-1
                
                for i in range(len(total_bbox)):
                    own_bbox=total_bbox[i]['bbox']
                    #print("own_box: ",i,own_bbox)
                    boxA=target
                    boxB=own_bbox
                    boxA = [int(x) for x in boxA]
                    boxB = [int(x) for x in boxB]
                    xA = max(boxA[0], boxB[0])
                    yA = max(boxA[1], boxB[1])
                    xB = min(boxA[2], boxB[2])
                    yB = min(boxA[3], boxB[3])

                    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

                    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
                    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
                    
                    iou = interArea / float(boxAArea + boxBArea - interArea)
                    if iou>max_iou:
                        max_iou=iou
                        pos=i
            if sym==1:
                break

        sym=0
        masks = []
        for num,seg in enumerate(self.panseg):
            #print(seg['file_name'])
            if file_name==seg['file_name']:
                sym=1
                seg_img=seg['pan_seg_file_name']
                #print(seg_img)
                seg_img=Image.open(self.image_root + seg_img)

                seg_map = self.convert_PIL_to_numpy(seg_img, "RGB")
                seg_map=self.rgb2id(seg_map)
                for i, s in enumerate(seg["segments_info"]):
                    masks.append((seg_map == s["id"]).tolist())
            if sym==1:
                break
        
        if sym==0:
            for num,seg in enumerate(self.panseg_val):
            #print(seg['file_name'])
                if file_name==seg['file_name']:
                    sym=1
                    seg_img=seg['pan_seg_file_name']
                    #print(seg_img)
                    seg_img=Image.open(self.image_root_val + seg_img)

                    seg_map = self.convert_PIL_to_numpy(seg_img, "RGB")
                    seg_map=self.rgb2id(seg_map)
                    for i, s in enumerate(seg["segments_info"]):
                        masks.append((seg_map == s["id"]).tolist())
                if sym==1:
                    break
        ori_masks=np.array(ori_masks)
        masks=np.array(masks)
        #masks=np.squeeze(masks,axis=0)  
        
        
        # target_list = np.zeros(self.obj_num)
        # target_list[pos]=1
        tokens, att_mask = self.tokenizer(sentence, **self.kwargs) # tokens, att_masks
        # pos1 = seq[-1]
        # pos2 = seq[-2]
        pic = self.img_dict[img_id+'.json']
        rel = self.rel_dict[str(index)]
        # rel = self.padding(item, rel, pos1, pos2)

        np_pic = np.array(pic).astype(np.float32)
        np_rel = np.array(rel).astype(np.float32)

        pic1 = torch.tensor(np_pic)
        rel1 = torch.tensor(np_rel)

        # list_p = list(pic1)
        # list_r = list(rel1)
        # res = [self.rel2id[item['relation']]] + seq + list_p + list_r
        return file_name,sentence,ori_masks,masks,index_pos,bbox,target,pos, tokens.squeeze(dim=0), att_mask.squeeze(dim=0), pic1, rel1 # label, seq1, seq2, ...,pic

    def convert_PIL_to_numpy(self,image, format):
        """
        Convert PIL image to numpy array of target format.

        Args:
            image (PIL.Image): a PIL image
            format (str): the format of output image

        Returns:
            (np.ndarray): also see `read_image`
        """
        if format is not None:
            # PIL only supports RGB, so convert to RGB and flip channels over below
            conversion_format = format
            if format in ["BGR", "YUV-BT.601"]:
                conversion_format = "RGB"
            image = image.convert(conversion_format)
        image = np.asarray(image)
        # PIL squeezes out the channel dimension for "L", so make it HWC
        if format == "L":
            image = np.expand_dims(image, -1)

        # handle formats not supported by PIL
        elif format == "BGR":
            # flip channels if needed
            image = image[:, :, ::-1]
        elif format == "YUV-BT.601":
            image = image / 255.0
            image = np.dot(image, np.array(_M_RGB2YUV).T)

        return image

    def rgb2id(self,color):
        if isinstance(color, np.ndarray) and len(color.shape) == 3:
            if color.dtype == np.uint8:
                color = color.astype(np.int32)
            return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
        return int(color[0] + 256 * color[1] + 256 * 256 * color[2])
    
    def bb_intersection_over_union(self,total_bbox):
        boxA=total_bbox[0]
        boxB=total_bbox[1]
        boxA = [int(x) for x in boxA]
        boxB = [int(x) for x in boxB]

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        
        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou

    def padding(self, item, rel, pos1, pos2):
        pos_head = item['h']['pos']
        pos_tail = item['t']['pos']

        pos_min = pos_head
        pos_max = pos_tail
        if pos_head[0] > pos_tail[0]:
            pos_min = pos_tail
            pos_max = pos_head
            rev = True
        else:
            rev = False
        if not rev:
            pos1_len = pos_min[1] - pos_min[0]
            pos2_len = pos_max[1] - pos_max[0]
        else:
            pos2_len = pos_min[1] - pos_min[0]
            pos1_len = pos_max[1] - pos_max[0]
        pad_pos = [0, pos1, pos1 + pos1_len + 1, pos2, pos2_len + pos2 + 1]
        rel_zero_list = list(np.zeros([self.obj_num]))
        for pos in pad_pos:
            rel.insert(pos, rel_zero_list)
        rel = rel[:self.length]
        return rel

    def collate_fn(data):
        data = list(zip(*data))
        labels = data[0]
        seqs = data[1:]
        batch_labels = torch.tensor(labels).long()  # (B)
        batch_seqs = []
        for seq in seqs:
            batch_seqs.append(torch.cat(seq, 0))  # (B, L)
        return [batch_labels] + batch_seqs

    def eval(self, pred_result, use_name=False):
        """
        Args:
            pred_result: a list of predicted label (id)
                Make sure that the `shuffle` param is set to `False` when getting the loader.
            use_name: if True, `pred_result` contains predicted relation names instead of ids
        Return:
            {'acc': xx}
        """
        correct = 0
        total = len(self.data)
        correct_positive = 0
        pred_positive = 0
        gold_positive = 0

        neg = -1
        for name in ['NA', 'na', 'no_relation', 'Other', 'Others', 'none', 'None']:
            if name in self.rel2id:
                if use_name:
                    neg = name
                else:
                    neg = self.rel2id[name]
                break
        for i in range(total):
            if use_name:
                golden = self.data[i]['relation']
            else:
                golden = self.rel2id[self.data[i]['relation']]

            if golden == pred_result[i]:
                correct += 1
                if golden != neg:
                    correct_positive += 1
            if golden != neg:
                gold_positive += 1
            if pred_result[i] != neg:
                pred_positive += 1
        acc = float(correct) / float(total)
        try:
            micro_p = float(correct_positive) / float(pred_positive)
        except:
            micro_p = 0
        try:
            micro_r = float(correct_positive) / float(gold_positive)
        except:
            micro_r = 0
        try:
            micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)
        except:
            micro_f1 = 0

        result = {'acc': acc, 'micro_p': micro_p, 'micro_r': micro_r, 'micro_f1': micro_f1}
        logging.info('Evaluation result: {}.'.format(result))
        return result


def SentenceRELoader_t(text_path, rel_path, pic_path, rel2id, tokenizer, batch_size,
                     shuffle, num_workers=1, **kwargs):
    dataset = SentenceREDataset_t(text_path=text_path, rel_path=rel_path, pic_path=pic_path,
                                rel2id=rel2id,
                                tokenizer=tokenizer,
                                kwargs=kwargs)
    # data_loader = data.DataLoader(dataset=dataset,
    #                               batch_size=batch_size,
    #                               shuffle=shuffle,
    #                               pin_memory=True,
    #                               num_workers=num_workers,
    #                               collate_fn=collate_fn)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  pin_memory=True,
                                  num_workers=num_workers)
    return data_loader

def SentenceRELoader_v(text_path, rel_path, pic_path, rel2id, tokenizer, batch_size,
                     shuffle, num_workers=5,  **kwargs):
    dataset = SentenceREDataset_v(text_path=text_path, rel_path=rel_path, pic_path=pic_path,
                                rel2id=rel2id,
                                tokenizer=tokenizer,
                                kwargs=kwargs)
    # data_loader = data.DataLoader(dataset=dataset,
    #                               batch_size=batch_size,
    #                               shuffle=shuffle,
    #                               pin_memory=True,
    #                               num_workers=num_workers,
    #                               collate_fn=collate_fn)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  pin_memory=True,
                                  num_workers=num_workers)
    return data_loader


class BagREDataset(data.Dataset):
    """
    Bag-level relation extraction dataset. Note that relation of NA should be named as 'NA'.
    """

    def __init__(self, path, pic_path, rel2id, tokenizer, entpair_as_bag=False, bag_size=0, mode=None):
        """
        Args:
            path: path of the input file
            rel2id: dictionary of relation->id mapping
            tokenizer: function of tokenizing
            entpair_as_bag: if True, bags are constructed based on same
                entity pairs instead of same relation facts (ignoring 
                relation labels)
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.rel2id = rel2id
        self.entpair_as_bag = entpair_as_bag
        self.bag_size = bag_size

        # Load the file
        f = open(path, 'r', encoding='utf-8')
        self.data = []
        for line in f:
            line = line.rstrip()
            if len(line) > 0:
                self.data.append(eval(line))
        f.close()

        # Load the pic feature file
        self.list_img = []
        f1 = open(pic_path, encoding='UTF-8')
        js2 = f1.readlines()
        for line in js2:
            line.strip().replace('\\n', '')
            l1 = line.split()
            self.list_img.append(l1)
        f1.close()
        logging.info("Loaded picture feature dataset {} with {} obejects.".format(pic_path, len(self.list_img)))

        # Construct bag-level dataset (a bag contains instances sharing the same relation fact)
        if mode == None:
            self.weight = np.ones((len(self.rel2id)), dtype=np.float32)
            self.bag_scope = []
            self.name2id = {}
            self.bag_name = []
            self.facts = {}
            for idx, item in enumerate(self.data):
                fact = (item['h'], item['t'], item['relation'])
                if item['relation'] != 'NA':
                    self.facts[str(fact)] = 1
                if entpair_as_bag:
                    name = (item['h'], item['t'])
                else:
                    name = fact
                if str(name) not in self.name2id:
                    self.name2id[str(name)] = len(self.name2id)
                    self.bag_scope.append([])
                    self.bag_name.append(name)
                self.bag_scope[self.name2id[str(name)]].append(idx)
                self.weight[self.rel2id[item['relation']]] += 1.0
            self.weight = 1.0 / (self.weight ** 0.05)
            self.weight = torch.from_numpy(self.weight)
        else:
            pass

    def __len__(self):
        return len(self.bag_scope)

    def __getitem__(self, index):
        bag = self.bag_scope[index]
        if self.bag_size > 0:
            if self.bag_size <= len(bag):
                resize_bag = random.sample(bag, self.bag_size)
            else:
                resize_bag = bag + list(np.random.choice(bag, self.bag_size - len(bag)))
            bag = resize_bag

        seqs = None
        pics = None
        rel = self.rel2id[self.data[bag[0]]['relation']]
        for sent_id in bag:
            item = self.data[sent_id]
            pic = self.list_img[sent_id]
            np_pic = np.array(pic).astype(np.float32)
            pic1 = torch.tensor(np_pic)
            pic1 = pic1.unsqueeze(1)
            seq = list(self.tokenizer(item))
            list_p = list(pic1)
            if seqs is None:
                seqs = []
                pics = []
                for i in range(len(seq)):
                    seqs.append([])
                    pics.append([])
            for i in range(len(seq)):
                seqs[i].append(seq[i])
                pics[i].append(list_p[i])
        for i in range(len(seqs)):
            seqs[i] = torch.cat(seqs[i], 0)  # (n, L), n is the size of bag
            pics[i] = torch.cat(pics[i], 0)
        return [rel, self.bag_name[index], len(bag)] + seqs + pics

    def collate_fn(data):
        data = list(zip(*data))
        label, bag_name, count = data[:3]
        seqs = data[3:]
        for i in range(len(seqs)):
            seqs[i] = torch.cat(seqs[i], 0)  # (sumn, L)
            seqs[i] = seqs[i].expand(
                (torch.cuda.device_count() if torch.cuda.device_count() > 0 else 1,) + seqs[i].size())
        scope = []  # (B, 2)
        start = 0
        for c in count:
            scope.append((start, start + c))
            start += c
        assert (start == seqs[0].size(1))
        scope = torch.tensor(scope).long()
        label = torch.tensor(label).long()  # (B)
        return [label, bag_name, scope] + seqs

    def collate_bag_size_fn(data):
        data = list(zip(*data))
        label, bag_name, count = data[:3]
        seqs = data[3:]
        for i in range(len(seqs)):
            seqs[i] = torch.stack(seqs[i], 0)  # (batch, bag, L)
        scope = []  # (B, 2)
        start = 0
        for c in count:
            scope.append((start, start + c))
            start += c
        label = torch.tensor(label).long()  # (B)
        return [label, bag_name, scope] + seqs

    def eval(self, pred_result):
        """
        Args:
            pred_result: a list with dict {'entpair': (head_id, tail_id), 'relation': rel, 'score': score}.
                Note that relation of NA should be excluded.
        Return:
            {'prec': narray[...], 'rec': narray[...], 'mean_prec': xx, 'f1': xx, 'auc': xx}
                prec (precision) and rec (recall) are in micro style.
                prec (precision) and rec (recall) are sorted in the decreasing order of the score.
                f1 is the max f1 score of those precison-recall points
        """
        sorted_pred_result = sorted(pred_result, key=lambda x: x['score'], reverse=True)
        prec = []
        rec = []
        correct = 0
        total = len(self.facts)
        for i, item in enumerate(sorted_pred_result):
            pred = item['entpair'][0], item['entpair'][1], item['relation']
            if str(pred) in self.facts:
                correct += 1
            prec.append(float(correct) / float(i + 1))
            rec.append(float(correct) / float(total))
        auc = sklearn.metrics.auc(x=rec, y=prec)
        np_prec = np.array(prec)
        np_rec = np.array(rec)
        f1 = (2 * np_prec * np_rec / (np_prec + np_rec + 1e-20)).max()
        mean_prec = np_prec.mean()
        return {'micro_p': np_prec, 'micro_r': np_rec, 'micro_p_mean': mean_prec, 'micro_f1': f1, 'auc': auc}


def BagRELoader(path, pic_path, rel2id, tokenizer, batch_size,
                shuffle, entpair_as_bag=False, bag_size=0, num_workers=8,
                collate_fn=BagREDataset.collate_fn):
    if bag_size == 0:
        collate_fn = BagREDataset.collate_fn
    else:
        collate_fn = BagREDataset.collate_bag_size_fn
    dataset = BagREDataset(path, pic_path, rel2id, tokenizer, entpair_as_bag=entpair_as_bag, bag_size=bag_size)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    return data_loader
