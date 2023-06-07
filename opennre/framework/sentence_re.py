import os, logging, json
from tqdm import tqdm
import torch
from torch import nn, optim
from .data_loader import SentenceRELoader_t,SentenceRELoader_v
from .utils import AverageMeter
import numpy as np
from sklearn import metrics
import logging
from PIL import Image
from random import randint

panseg=json.load(open("./benchmark/ours/for_pan_seg.json","r"))

class SentenceRE(nn.Module):

    def __init__(self,
                 model,
                 train_path,
                 train_rel_path,
                 train_pic_path,
                 val_path,
                 val_rel_path,
                 val_pic_path,
                 test_path,
                 test_rel_path,
                 test_pic_path,
                 ckpt,
                 batchsize=32,
                 max_epoch=100,
                 lr=0.01,
                 weight_decay=1e-5,
                 warmup_step=300,
                 opt='sgd'):

        super().__init__()
        self.max_epoch = max_epoch
        # Load data
        if train_path != None:
            self.train_loader = SentenceRELoader_t(
                train_path,
                train_rel_path,
                train_pic_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batchsize,
                True)

        if val_path != None:
            self.val_loader = SentenceRELoader_v(
                val_path,
                val_rel_path,
                val_pic_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size=1,
                shuffle=False)

        # if test_path != None:
        #     self.test_loader = SentenceRELoader(
        #         test_path,
        #         test_rel_path,
        #         test_pic_path,
        #         model.rel2id,
        #         model.sentence_encoder.tokenize,
        #         batch_size,
        #         False
        #     )
        # Model
        #self.f = open('./results_log', 'a')        
        self.model = model
        #self.parallel_model = nn.DataParallel(self.model)
        # Criterion
        self.criterion = nn.CrossEntropyLoss()
        # Params and optimizer
        #params = self.parameters()
        params = model.parameters()
        self.lr = lr
        if opt == 'sgd':
            self.optimizer = optim.SGD(params, lr, weight_decay=weight_decay)
        elif opt == 'adam':
            self.optimizer = optim.Adam(params, lr, weight_decay=weight_decay)
        elif opt == 'adamw':  # Optimizer for BERT
            from transformers import AdamW
            params = list(self.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            grouped_params = [
                {
                    'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
                    'weight_decay': 0.01,
                    'lr': lr,
                    'ori_lr': lr
                },
                {
                    'params': [p for n, p in params if any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0,
                    'lr': lr,
                    'ori_lr': lr
                }
            ]
            #self.optimizer = AdamW(model.parameters(), lr=0.01,correct_bias=False)
            self.optimizer = AdamW(grouped_params, correct_bias=False)
        else:
            raise Exception("Invalid optimizer. Must be 'sgd' or 'adam' or 'adamw'.")
        # Warmup
        if warmup_step > 0:
            from transformers import get_linear_schedule_with_warmup
            training_steps = self.train_loader.dataset.__len__() // batchsize * self.max_epoch
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_step,
                                                             num_training_steps=training_steps)
        else:
            self.scheduler = None
        # Cuda
        if torch.cuda.is_available():
            self.cuda()
        # Ckpt
        self.ckpt = ckpt

    def train_model(self, metric='acc'):
        best_metric = 0
        global_step = 0
        # loader = [self.train_loader, self.val_loader]
        for epoch in range(self.max_epoch):
            self.train()
            logging.info("=== Epoch %d train ===" % epoch)
            avg_loss = AverageMeter()
            avg_acc = AverageMeter()
            avg_f1 = AverageMeter()
            t = tqdm(self.train_loader, ncols=110)
            total_loss=0
            if_con=0
            for iter, data in enumerate(t):
                symbol,bbox,target_list, tokens, att_mask, list_p, list_r = data

                
                for a,b in enumerate(symbol):
                    if b==False:
                        #print("WARNING!")
                        if_con+=1

                
                target_list=np.array(target_list)
                target_list=torch.from_numpy(target_list)

                target_list=target_list.cuda()
                #print(target_list)
                _,target_pos=target_list.max(1)


                #print(target_pos)
                #print("target_pos:  ",target_pos)
                tokens=tokens.cuda()
                att_mask=att_mask.cuda()
                list_p=list_p.cuda()
                list_r=list_r.cuda()

                logits = self.model(tokens, att_mask, list_p, list_r)
                #print(logits)
                target=target_list.long()
                loss = self.criterion(logits, target_pos)
                total_loss+=loss
                #loss = (loss-0.02).abs()+0.02

                score, pred = logits.max(1)  # (B)
                #print("pred :  ",pred)
                #acc = float(pred == target_pos)
                # acc = float((pred == target_pos).long().sum()) / label.size(0)
                #f1 = metrics.f1_score(pred.cpu(), label.cpu(), average='macro')
                # Log
                avg_loss.update(loss.item(), 1)
                #avg_acc.update(acc, 1)
                #avg_f1.update(f1, 1)
                #t.set_postfix(loss=avg_loss.avg, acc=avg_acc.avg, f1=avg_f1.avg)
                # Optimize
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                global_step += 1
            # Val
            # print("if_con:  ",if_con)
            print("TOTAL LOSS:  ",total_loss)
            logging.info("=== Epoch %d val ===" % epoch)
            #self.f.write('VAL: {}\n'.format(epoch))
            self.eval_model(self.val_loader,epoch)

            # result = self.eval_model(self.val_loader)
        #     logging.info('Metric {} current / best: {} / {}'.format(metric, result[metric], best_metric))
        #     if result[metric] > best_metric:
        #         logging.info("Best ckpt and saved.")
        #         folder_path = '/'.join(self.ckpt.split('/')[:-1])
        #         if not os.path.exists(folder_path):
        #             os.mkdir(folder_path)
        #         torch.save({'state_dict': self.model.state_dict()}, self.ckpt)
        #         best_metric = result[metric]
        #logging.info("Best %s on val set: %f" % (metric, best_metric))

    def eval_model(self, eval_loader,epoch):
        self.eval()
        avg_acc = AverageMeter()
        avg_loss = AverageMeter()
        pred_result = []
        with torch.no_grad():
            total_target=[]
            total_result=[]
            total_iou_bbox=[]
            total_iou_mask=[]
            low_confidence=0
            num_3=0
            num_4=0
            num_5=0
            num_6=0
            num_7=0

            performance=[]
            t = tqdm(eval_loader, ncols=110)
            number_sample=len(t)
            for iter, data in enumerate(t):
                file_name,sentence,ori_masks,masks,target_p,bbox,target_b,pos, tokens, att_mask, list_p, list_r = data

                ori_masks=np.squeeze(ori_masks,axis=0)
                masks=np.squeeze(masks,axis=0)
                len_masks=len(masks)
                bbox=torch.squeeze(bbox)
                #print(bbox.shape)


                target_bbox=target_b
                total_target.append(target_p)
                tokens=tokens.cuda()
                att_mask=att_mask.cuda()
                list_p=list_p.cuda()
                list_r=list_r.cuda()


                logits = self.model(tokens, att_mask, list_p, list_r)
                #print(logits)
                #loss = self.criterion(logits, target_pos)
                score, pred = logits.max(1)  # (B)
                #print(len_masks)
                # if target_pos+1>len_masks:
                #     print("FALSE")
                #     continue
                if pred+1>len_masks:
                    pred=torch.tensor([randint(0,len_masks-1)])

                max_area=-1
                new_pos=-1
                if score<0.5:
                    
                    for i,mask in enumerate(masks):
                        mask=np.array(mask)
                        mask=1*mask
                        area=mask.sum()
                        if area>max_area:
                            max_area=area
                            new_pos=i
                if new_pos==-1:
                    pred_bbox=bbox[pred,:]
                    maskB=masks[pred,:,:]
                    pred_bbox=pred_bbox[0]
                elif new_pos<10:
                    #print("LOW CONFIDENCE")
                    low_confidence+=1
                    pred_bbox=bbox[new_pos,:]
                    maskB=masks[new_pos,:,:]
                elif new_pos>=10:
                    low_confidence+=1
                    maskB=masks[new_pos,:,:]
                    pred_bbox=bbox[torch.tensor([randint(0,9)]),:]
                    pred_bbox=pred_bbox[0]
                #print(pred_bbox.shape)    
                

                #print(target_bbox)
                #print(pred_bbox)  
                 
                #print(masks.shape) 
                  
                # print(bbox.shape)
                # print(masks.shape)
                maskA=ori_masks[pos,:,:]
                
                
                maskA=torch.squeeze(maskA)
                maskB=torch.squeeze(maskB)
                
                #print(type(maskA))
                #print(type(maskB))
                maskA=np.array(maskA)
                maskB=np.array(maskB)
                maskA=1*maskA
                maskB=1*maskB
                # print(maskA.shape)
                # print(maskB.shape)
                
                area1 = maskA.sum()
                #print("A1:  ",area1)
            
                area2 = maskB.sum()
                #print("A2:  ",area2)
                inter = ((maskA+maskB)==2).sum()
                #print("INTER:  ",inter)
                iou = inter/(area1+area2-inter)
                #print("IOU:  ",iou)
                total_iou_mask.append(iou)
                result={
                    'file_name':file_name,
                    'sentence':sentence,
                    'IoU':iou,
                    'pred':np.array(pred.cpu()).tolist()

                }
                performance.append(result)
                if iou>0.3:
                    num_3+=1
                if iou>0.4:
                    num_4+=1
                if iou>0.5:
                    num_5+=1
                if iou>0.6:
                    num_6+=1
                if iou>0.7:
                    num_7+=1
                # print(pred)

                #print(pred_bbox)
                boxA=target_bbox
                boxB=pred_bbox
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
                total_iou_bbox.append(iou)



                total_result.append(pred.cpu())
        #         # Save result
        #         for i in range(pred.size(0)):
        #             pred_result.append(pred[i].item())
        #         # Log
        #         # acc = float((pred == label).long().sum()) / label.size(0)
        #         acc = float(pred == label)
        #         avg_acc.update(acc, pred.size(0))
        #         avg_loss.update(loss.item(), 1)
        #         t.set_postfix(loss=avg_loss.avg,acc=avg_acc.avg)
        # result = eval_loader.dataset.eval(pred_result)
        total_target=np.array(total_target)
        total_result=np.array(total_result)
        total_iou_bbox=np.array(total_iou_bbox)
        total_iou_mask=np.array(total_iou_mask)
        
        # print("TARGET: ",total_target)
        # print("RESULT: ",total_result)
        accuracy=((total_target==total_result).sum())/len(total_target)
        m_Iou_b=sum(total_iou_bbox)/len(total_iou_bbox)
        m_Iou_m=sum(total_iou_mask)/len(total_iou_mask)
        #print(accuracy)
        #self.f.write('ACC: {}\n'.format(accuracy))
        logging.info('LOW CONFIDENCE: %f' %(low_confidence))
        logging.info('ACC: %f' %(accuracy))
        logging.info('m_IoU_b: %f' %(m_Iou_b))
        logging.info('m_IoU_m: %f' %(m_Iou_m))
        logging.info('P@0.3: %f' %(num_3/number_sample))
        logging.info('P@0.4: %f' %(num_4/number_sample))
        logging.info('P@0.5: %f' %(num_5/number_sample))
        logging.info('P@0.6: %f' %(num_6/number_sample))
        logging.info('P@0.7: %f' %(num_7/number_sample))

        result={
            'ACC':accuracy,
            'm_IoU_b':m_Iou_b,
            'm_IoU_m':m_Iou_m,
            'P@0.3':num_3/number_sample,
            'P@0.4':num_4/number_sample,
            'P@0.5':num_5/number_sample,
            'P@0.6':num_6/number_sample,
            'P@0.7':num_7/number_sample,
        }
        performance.append(result)

        with open('./logfeature/epoch_'+str(epoch)+'.json', 'w') as outfile:
            json.dump(performance, outfile, default=str)
        #return result

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
