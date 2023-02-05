import math
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from skimage import color

from mmdet.core import bbox_target
from mmdet.core.utils import multi_apply
from mmdet.ops import ConvModule
from mmdet.models.losses import accuracy
from ..registry import HEADS
# from ..utils import multi_apply
from .bbox_head import BBoxHead
from mmdet.models.losses.color_regression_loss import ColorLoss


@HEADS.register_module
class ConvFCBBoxHead(BBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

                                /-> cls convs -> cls fcs -> cls
    shared convs -> shared fcs
                                \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 *args,
                 **kwargs):
        super(ConvFCBBoxHead, self).__init__(*args, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # add shared convs and fcs 共享分支
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch 分类分支
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch 回归分支
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            self.fc_cls = nn.Linear(self.cls_last_dim, self.num_classes)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (last_layer_dim
                                    if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(conv_in_channels,
                               self.conv_out_channels,
                               3,
                               padding=1,
                               conv_cfg=self.conv_cfg,
                               norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (last_layer_dim
                                  if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def init_weights(self):
        super(ConvFCBBoxHead, self).init_weights()
        # conv layers are already initialized by ConvModule
        for module_list in [self.shared_fcs, self.cls_fcs, self.reg_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x
        # 分类分支
        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))
        # 回归分支
        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred


@HEADS.register_module
class SharedFCBBoxHead(ConvFCBBoxHead):

    def __init__(self, num_fcs=2, fc_out_channels=1024, *args, **kwargs):
        assert num_fcs >= 1
        super(SharedFCBBoxHead, self).__init__(num_shared_convs=0,
                                               num_shared_fcs=num_fcs,
                                               num_cls_convs=0,
                                               num_cls_fcs=0,
                                               num_reg_convs=0,
                                               num_reg_fcs=0,
                                               fc_out_channels=fc_out_channels,
                                               *args,
                                               **kwargs)


@HEADS.register_module
class ColorSimBBoxHead(SharedFCBBoxHead):
    """添加颜色相似度分支
        
                                /-> cls convs -> cls fcs -> cls
    shared convs -> shared fcs
                                              /-> reg fcs -> reg
                                \-> reg convs  
                                              \-> color_sim convs -> color_smi   # 颜色相似度分支                         
    """

    def __init__(self,
                 num_color_convs=0,
                 num_color_fcs=0,
                 with_color=True,
                 *args,
                 **kwargs):
        super(ColorSimBBoxHead, self).__init__(*args, **kwargs)
        self.num_color_convs = num_color_convs
        self.num_color_fcs = num_color_fcs
        self.with_color = with_color
        # # 添加分类
        # self.color_conv = ConvModule(self.reg_last_dim,
        #                              1,
        #                              3,
        #                              stride=1,
        #                              padding=1,
        #                              conv_cfg=self.conv_cfg,
        #                              norm_cfg=self.norm_cfg)

        # 颜色相似度分支
        self.color_convs, self.color_fcs, self.color_last_dim = \
            self._add_conv_fc_branch(
                self.num_color_convs, self.num_color_fcs, self.shared_out_channels)

        if self.with_color:
            self.color_pred = nn.Linear(self.color_last_dim, self.num_classes)
        self.color_loss = ColorLoss(loss_weight=1.0)

    def forward(self, x):
        # 共享分支
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x  # 分类分支的输入
        x_reg = x  # 回归分支的输入
        # 分类分支
        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))
        # 回归分支
        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        x_color = x_reg  # 颜色分支的输入
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        # 颜色相似度分支
        for conv in self.color_convs:
            x_color = conv(x_color)
        if x_color.dim() > 2:
            if self.with_avg_pool:
                x_color = self.avg_pool(x_color)
            x_color = x_color.flatten(1)
        for fc in self.color_fcs:
            x_color = self.relu(fc(x_color))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        # 使用线性层处理颜色相似度
        color_pred = self.color_pred(x_color) if self.with_color else None
        # 返回颜色相似度
        return cls_score, bbox_pred, color_pred

    def get_target(self, sampling_results, gt_bboxes, gt_labels,
                   rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        neg_proposals = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
        reg_classes = 1 if self.reg_class_agnostic else self.num_classes
        cls_reg_targets = bbox_target(pos_proposals,
                                      neg_proposals,
                                      pos_gt_bboxes,
                                      pos_gt_labels,
                                      rcnn_train_cfg,
                                      reg_classes,
                                      target_means=self.target_means,
                                      target_stds=self.target_stds)
        # 获取颜色相似度分支的目标
        # color_targets = self.color_target(pos_proposals, pos_gt_bboxes)
        pos_index = cls_reg_targets[0]
        pos_index[pos_index > 0] = 1  # 标识正样本
        return cls_reg_targets, pos_proposals, pos_gt_bboxes, pos_index

    def color_target(self, img_list, pos_proposals_list, pos_gt_bboxes_list):
        """ 对所有图片中的roi计算颜色相似度标签 """
        assert len(img_list) == len(pos_proposals_list) == len(
            pos_gt_bboxes_list)
        labels = multi_apply(self.color_target_single, img_list,
                             pos_proposals_list, pos_gt_bboxes_list)
        num_img = len(img_list)
        # 对标签进行合并
        label_concat = []
        for label in labels[0]:
            label_concat += label
        return label_concat

    def color_target_single(self, img, pos_bboxes, pos_gt_bboxes):
        """ 获取单张图片中roi的颜色相似度gt """
        num_pos = pos_bboxes.size(0)
        if num_pos > 0:
            # 将图片转换为LAB颜色空间
            # images_lab = color.rgb2lab(img.byte().permute(1, 2,0).cpu().numpy())
            # images_lab = torch.as_tensor(img,
            #                              device=torch.device('cuda'),
            #                              dtype=torch.float32)
            # images_lab = images_lab.permute(2, 0, 1)[None]
            # 解析roi坐标
            roi_x_min_0 = pos_bboxes[:, 0]
            roi_y_min_0 = pos_bboxes[:, 1]
            roi_x_max_0 = pos_bboxes[:, 2]
            roi_y_max_0 = pos_bboxes[:, 3]
            # 解析roi_gt坐标
            gt_x_min_0 = pos_gt_bboxes[:, 0]
            gt_y_min_0 = pos_gt_bboxes[:, 1]
            gt_x_max_0 = pos_gt_bboxes[:, 2]
            gt_y_max_0 = pos_gt_bboxes[:, 3]
            # 取出roi对应的原图中的像素
            map_roi = [
                img[:,
                    int(roi_y_min_0[i]):int(roi_y_max_0[i]),
                    int(roi_x_min_0[i]):int(roi_x_max_0[i])]
                for i in range(num_pos)
            ]
            # 取出gt对应的原图中的像素
            map_gt = [
                img[:,
                    int(gt_y_min_0[i]):int(gt_y_max_0[i]),
                    int(gt_x_min_0[i]):int(gt_x_max_0[i])]
                for i in range(num_pos)
            ]
            # 单张图片中计算颜色相似度标签
            labels_single = self.get_images_color_similarity(map_roi, map_gt)

        return labels_single

    def get_images_color_similarity(self, map_roi_list, map_gt_list):
        """ 计算roi颜色相似度 """

        similarity = multi_apply(self.get_images_color_similarity_single,
                                 map_roi_list, map_gt_list)

        return similarity

    def colorValMapping(self, val):
        mapVal = 0
        if val > 223:
            # [224 ~ 255]
            mapVal = 7
        elif val > 191:
            # [192 ~ 223]
            mapVal = 6
        elif val > 159:
            # [160 ~ 191]
            mapVal = 5
        elif val > 127:
            # [128 ~ 159]
            mapVal = 4
        elif val > 95:
            # [96 ~ 127]
            mapVal = 3
        elif val > 63:
            # [64 ~ 95]
            mapVal = 2
        elif val > 31:
            # [32 ~ 63]
            mapVal = 1
        else:
            # [0 ~ 31]
            mapVal = 0
        return mapVal

    def compImageFeature(self,img,img_feature):
        index = 0
        heigh, width = img.size()[-2:]
        for i in range(0, heigh):
            for j in range(width):
                map_r = self.colorValMapping(img[0,i, j])
                map_g = self.colorValMapping(img[1,i, j])
                map_b = self.colorValMapping(img[2,i, j])
                index=map_b*64+map_g*8+map_r
                img_feature[index]+=1


    def de_norm(self,img,mean=[127.5, 127.5, 127.5],std=[127.5, 127.5, 127.5]):
        channel_mean = torch.tensor(mean)
        channel_std = torch.tensor(std)

        MEAN = [-mean/std for mean, std in zip(channel_mean, channel_std)]
        STD = [1/std for std in channel_std]
        denormalizer = transforms.Normalize(mean=MEAN, std=STD)(img)

        return denormalizer

    def get_hsv_hist_feature(self,img):
        # 获取图像hsv直方图特征，返回h,s,v三个维度各自的直方图
        hsv = color.rgb2hsv(img.byte().permute(1, 2,0).cpu().numpy())
        # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        arr_h=hsv[:,:,0].flatten()
        arr_s=hsv[:,:,1].flatten()
        arr_v=hsv[:,:,2].flatten()
        nh, bins, patches= plt.hist(arr_h, bins=256,   density=True)
        ns, bins, patches = plt.hist(arr_s, bins=256,  density=True)
        nv, bins, patches = plt.hist(arr_v, bins=256,  density=True)
        return [nh,ns,nv]

    def calculate(self,image1, image2):
        # 灰度直方图算法
        # 计算单通道的直方图的相似值
        hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
        # nh, bins, patches = plt.hist(image1, bins=256,   density=True)
        hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
        # 计算直方图的重合度
        degree = 0
        for i in range(len(hist1)):
            if hist1[i] != hist2[i]:
                degree = degree + (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
            else:
                degree = degree + 1
        degree = degree / len(hist1)
        return degree

    def get_images_color_similarity_single(self, map_roi, map_gt):
        # 计算2张图片的颜色特征的相似度
        map_roi = self.de_norm(map_roi)
        map_gt = self.de_norm(map_gt)


        sub_image1 = cv2.split(map_roi.permute(1, 2,0).cpu().numpy())
        sub_image2 = cv2.split(map_gt.permute(1, 2,0).cpu().numpy())
        sub_data = 0
        for im1, im2 in zip(sub_image1, sub_image2):
            sub_data += self.calculate(im1, im2)
        sub_data = sub_data / 3
        if isinstance(sub_data,np.ndarray):
            sub_data=sub_data[0]
        return (sub_data,)

        # value=0
        # feature1=self.get_hsv_hist_feature(map_roi)
        # feature2=self.get_hsv_hist_feature(map_gt)
        # for i in range(2):
        #     count1=feature1[i]
        #     count2=feature2[i]
        #     Sum1=sum(count1);Sum2=sum(count2)
        #     Sumup = [math.sqrt(a*b) for a,b in zip(count1,count2)]
        #     SumDown = math.sqrt(Sum1*Sum2)
        #     Sumup = sum(Sumup)
        #     HistDist=1-math.sqrt(1-Sumup/SumDown)
        #     value+=HistDist
            
        # similarity = value / 3
        # return (similarity,)

        # img1_feature = [0 for i in range(0, 512)]
        # img2_feature = [0 for i in range(0, 512)]

        # map_roi = self.de_norm(map_roi)
        # self.compImageFeature(map_roi, img1_feature)
        # map_gt = self.de_norm(map_gt)
        # self.compImageFeature(map_gt, img2_feature)

        # sum_square0 = 0.0
        # sum_square1 = 0.0
        # sum_multiply = 0.0

        # for i in range(0, 512):
        #     sum_square0 += img1_feature[i] * img1_feature[i]
        #     sum_square1 += img2_feature[i] * img2_feature[i]
        #     sum_multiply += img1_feature[i] * img2_feature[i]

        # similarity = sum_multiply / (np.sqrt(sum_square0) *
        #                              np.sqrt(sum_square1))
        # """ 获取单个roi的颜色相似度 """
        # if map_roi.size() != map_gt.size():
        #     # 如果具有不同的size，则进行中心剪裁
        #     map_roi, map_gt = self.center_crop(map_roi, map_gt)
        # # 计算相关度
        # diff = map_gt - map_roi
        # # 按照通道计算颜色相似度
        # similarity = torch.exp(-torch.norm(diff, dim=0) * 0.5)
        # # 将所有像素点的平均值作为最终的颜色相似度
        # similarity = torch.tensor(similarity).cpu().item()
        # return (similarity, )

    def center_crop(self, img_1, img_2):
        """ 将两张图片进行中心剪裁，使两张图片具有相同的大小 """
        # 取图片宽高
        h_img_1, w_img_1 = img_1.size()[-2:]
        h_img_2, w_img_2 = img_2.size()[-2:]
        # 取图片中心点坐标
        mid_img_1_h = int(h_img_1 / 2)
        mid_img_1_w = int(w_img_1 / 2)
        mid_img_2_h = int(h_img_2 / 2)
        mid_img_2_w = int(w_img_2 / 2)
        # 取需要剪裁的宽高
        w_crop = w_img_1 if w_img_1 < w_img_2 else w_img_2
        h_crop = h_img_1 if h_img_1 < h_img_2 else h_img_2

        w_point = int(w_crop / 2)
        h_point = int(h_crop / 2)
        #对图片1进行中心剪裁
        top = mid_img_1_h - h_point
        bottom = mid_img_1_h + h_point
        left = mid_img_1_w - w_point
        right = mid_img_1_w + w_point
        img_1 = img_1[:, top:bottom, left:right]
        # 对图片2进行中心剪裁
        top = mid_img_2_h - h_point
        bottom = mid_img_2_h + h_point
        left = mid_img_2_w - w_point
        right = mid_img_2_w + w_point
        img_2 = img_2[:, top:bottom, left:right]

        return img_1, img_2

    def loss(self,
             cls_score,
             bbox_pred,
             color_pred,
             pos_index,
             color_target,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        """ 计算损失：分类+回归+颜色相似度 """
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                losses['loss_cls'] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            pos_inds = labels > 0
            if pos_inds.any():
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
        # 计算颜色相似度损失
        if color_pred is not None:
            color_pred = color_pred[pos_index > 0]  # 筛选正样本
            color_pred = torch.mean(color_pred, 1)  # 求平均作为最后的的颜色相似度的预测
            device = torch.device('cuda')
            color_target = torch.tensor(color_target, device=device)
            losses['color_loss'] = self.color_loss(color_pred, color_target)
        return losses
