import numpy as np
from torch import nn

from .cascade_rcnn import CascadeRCNN
import torch

from ..registry import DETECTORS

from mmdet.core import (bbox2roi, build_assigner, build_sampler)
from matplotlib import pyplot as plt
import torch.nn.functional as F
from math import log
from mmdet.core import (bbox2result, merge_aug_masks)
from mmdet.models.losses.dice_loss import DiceLoss
from PIL import Image


class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class EAM(nn.Module):
    def __init__(self):
        super(EAM, self).__init__()
        self.reduce1 = Conv1x1(256, 64)
        self.reduce4 = Conv1x1(2048, 256)
        self.block = nn.Sequential(
            ConvBNR(256 + 64, 256, 3),
            ConvBNR(256, 256, 3),
            nn.Conv2d(256, 1, 1))

    def forward(self, x4, x1):
        size = x1.size()[2:]
        x1 = self.reduce1(x1)
        x4 = self.reduce4(x4)
        x4 = F.interpolate(x4, size, mode='bilinear', align_corners=False)
        out = torch.cat((x4, x1), dim=1)
        out = self.block(out)

        return out


class EFM(nn.Module):
    """边缘增强模块"""

    def __init__(self, channel):
        super(EFM, self).__init__()
        t = int(abs((log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv2d = ConvBNR(channel, channel, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, c, att):
        if c.size() != att.size():
            att = F.interpolate(att, c.size()[2:], mode='bilinear', align_corners=False)
        x = c * att + c
        x = self.conv2d(x)
        wei = self.avg_pool(x)
        wei = self.conv1d(wei.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        wei = self.sigmoid(wei)
        x = x * wei

        return x


class CAM(nn.Module):
    def __init__(self, hchannel, channel):
        super(CAM, self).__init__()
        self.conv1_1 = Conv1x1(hchannel + channel, channel)
        self.conv3_1 = ConvBNR(channel // 4, channel // 4, 3)
        self.dconv5_1 = ConvBNR(channel // 4, channel // 4, 3, dilation=2)
        self.dconv7_1 = ConvBNR(channel // 4, channel // 4, 3, dilation=3)
        self.dconv9_1 = ConvBNR(channel // 4, channel // 4, 3, dilation=4)
        self.conv1_2 = Conv1x1(channel, channel)
        self.conv3_3 = ConvBNR(channel, channel, 3)

    def forward(self, lf, hf):
        if lf.size()[2:] != hf.size()[2:]:
            hf = F.interpolate(hf, size=lf.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((lf, hf), dim=1)
        x = self.conv1_1(x)
        xc = torch.chunk(x, 4, dim=1)
        x0 = self.conv3_1(xc[0] + xc[1])
        x1 = self.dconv5_1(xc[1] + x0 + xc[2])
        x2 = self.dconv7_1(xc[2] + x1 + xc[3])
        x3 = self.dconv9_1(xc[3] + x2)
        xx = self.conv1_2(torch.cat((x0, x1, x2, x3), dim=1))
        x = self.conv3_3(x + xx)

        return x


@DETECTORS.register_module
class BGCascadeRCNN(CascadeRCNN):
    """添加了边缘模块的Cascade Mask RCNN"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.eam = EAM()
        self.efm1 = EFM(256)
        self.efm2 = EFM(512)
        self.efm3 = EFM(1024)
        self.efm4 = EFM(2048)
        # self.reduce1 = Conv1x1(256, 64)
        # self.reduce2 = Conv1x1(512, 128)
        # self.reduce3 = Conv1x1(1024, 256)
        # self.reduce4 = Conv1x1(2048, 256)
        # self.cam1 = CAM(128, 64)
        # self.cam2 = CAM(256, 128)
        # self.cam3 = CAM(256, 256)

        self.loss = DiceLoss(loss_weight=3.0)

    def edge_enhance(self, input):
        """边缘模块"""
        x1, x2, x3, x4 = input
        edge = self.eam(x4, x1)  # 预测边缘
        edge_att = torch.sigmoid(edge)
        # 边缘特征增强
        x1a = self.efm1(x1, edge_att)
        x2a = self.efm2(x2, edge_att)
        x3a = self.efm3(x3, edge_att)
        x4a = self.efm4(x4, edge_att)
        # 边缘融合
        # x1r = self.reduce1(x1a)
        # x2r = self.reduce2(x2a)
        # x3r = self.reduce3(x3a)
        # x4r = self.reduce4(x4a)
        # x34 = self.cam3(x3r, x4r)
        # x234 = self.cam2(x2r, x34)
        # x1234 = self.cam1(x1r, x234)

        return edge_att, (x1a, x2a, x3a, x4a)

    def extract_feat(self, img):
        x = self.backbone(img)
        return x

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      edge_map,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None):

        x = self.extract_feat(img)  # 使用backbone 提取特征
        # x = self.enhance_feature(x)

        edge_att, x = self.edge_enhance(x)  # 使用边缘模块进行增强

        if self.with_neck:  # 使用FPN进行特征融合
            x = self.neck(x)

        # 失计算边缘DICE损失
        oe = F.interpolate(edge_att, edge_map.size()[2:], mode='bilinear', align_corners=False)
        # edge_losse = self.dice_loss(oe, edge_map)
        edge_losse = dict(edge_losse=self.loss(oe, edge_map))

        losses = dict()

        losses.update(edge_losse)
        # 将边缘增强后的模块输入PRN中进行proposals
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_metas,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(*rpn_loss_inputs,
                                            gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_metas, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        for i in range(self.num_stages):
            self.current_stage = i
            rcnn_train_cfg = self.train_cfg.rcnn[i]
            lw = self.train_cfg.stage_loss_weights[i]

            # assign gts and sample proposals
            sampling_results = []
            if self.with_bbox or self.with_mask:
                bbox_assigner = build_assigner(rcnn_train_cfg.assigner)
                bbox_sampler = build_sampler(rcnn_train_cfg.sampler,
                                             context=self)
                num_imgs = img.size(0)
                if gt_bboxes_ignore is None:
                    gt_bboxes_ignore = [None for _ in range(num_imgs)]

                for j in range(num_imgs):
                    assign_result = bbox_assigner.assign(
                        proposal_list[j], gt_bboxes[j], gt_bboxes_ignore[j],
                        gt_labels[j])
                    sampling_result = bbox_sampler.sample(
                        assign_result,
                        proposal_list[j],
                        gt_bboxes[j],
                        gt_labels[j],
                        feats=[lvl_feat[j][None] for lvl_feat in x])
                    sampling_results.append(sampling_result)

            # bbox head forward and loss
            bbox_roi_extractor = self.bbox_roi_extractor[i]
            bbox_head = self.bbox_head[i]

            rois = bbox2roi([res.bboxes for res in sampling_results])
            # mask = rois[:]
            # roi_img0 = rois[rois[:, 0] == 0][:, 1:5].int().cpu().numpy()  # 图片0中的rois
            # x_min_0 = roi_img0[:, 0]
            # y_min_0 = roi_img0[:, 1]
            # x_max_0 = roi_img0[:, 2]
            # y_max_0 = roi_img0[:, 3]
            # img = img[0, :, :, :]
            # 取出roi对应的原图中的像素
            # map_roi = [img[:, y_min_0[i]:y_max_0[i], x_min_0[i]:x_max_0[i]] for i in range(roi_img0.shape[0])]

            # 找到roi对应的gt

            # 取出roi对应的gt在原图中的像素

            #
            # roi_img1 = rois[rois[:, 0] == 1][:, 1:5].int()  # 图片1中的rois
            # x_min_1 = roi_img1[:, 0]
            # y_min_1 = roi_img1[:, 1]
            # x_max_1 = roi_img1[:, 2]
            # y_max_1 = roi_img1[:, 3]
            # # feat = img[0, :, int(roi_img0[:, 0]):int(roi_img0[:, 2]), int(roi_img0[:, 1]):int(roi_img0[:, 3])]

            if len(rois) == 0:
                # If there are no predicted and/or truth boxes, then we cannot
                # compute head / mask losses
                continue
            # 根据rois提取特征
            bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                            rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            # 使用bbox head进行分类和回归
            # cls_score, bbox_pred = bbox_head(bbox_feats)
            cls_score, bbox_pred, color_pred = bbox_head(bbox_feats)
            # 获取ground-truth
            bbox_targets, pos_proposals, pos_gt_bboxes, pos_index = bbox_head.get_target(
                sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg)
            # 获取颜色相似度标签
            color_target = bbox_head.color_target(img, pos_proposals,
                                                  pos_gt_bboxes)
            # 计算损失
            loss_bbox = bbox_head.loss(cls_score, bbox_pred, color_pred,
                                       pos_index, color_target, *bbox_targets)
            # img_shape = img_metas[0]['img_shape']
            # ori_shape = img_metas[0]['ori_shape']
            # scale_factor = img_metas[0]['scale_factor']
            # det_bboxes, det_labels = self.bbox_head[-1].get_det_bboxes(
            #     rois,
            #     cls_score,
            #     bbox_pred,
            #     img_shape,
            #     scale_factor,
            #     rescale=False)
            for name, value in loss_bbox.items():
                losses['s{}.{}'.format(
                    i, name)] = (value * lw if 'loss' in name else value)

            # mask head forward and loss
            if self.with_mask:
                if not self.share_roi_extractor:
                    mask_roi_extractor = self.mask_roi_extractor[i]
                    pos_rois = bbox2roi(
                        [res.pos_bboxes for res in sampling_results])
                    mask_feats = mask_roi_extractor(
                        x[:mask_roi_extractor.num_inputs], pos_rois)
                    if self.with_shared_head:
                        mask_feats = self.shared_head(mask_feats)
                else:
                    # reuse positive bbox feats
                    pos_inds = []
                    device = bbox_feats.device
                    for res in sampling_results:
                        pos_inds.append(
                            torch.ones(res.pos_bboxes.shape[0],
                                       device=device,
                                       dtype=torch.uint8))
                        pos_inds.append(
                            torch.zeros(res.neg_bboxes.shape[0],
                                        device=device,
                                        dtype=torch.uint8))
                    pos_inds = torch.cat(pos_inds)
                    mask_feats = bbox_feats[pos_inds.type(torch.bool)]
                mask_head = self.mask_head[i]
                mask_pred = mask_head(mask_feats)
                mask_targets = mask_head.get_target(sampling_results, gt_masks,
                                                    rcnn_train_cfg)
                pos_labels = torch.cat(
                    [res.pos_gt_labels for res in sampling_results])
                loss_mask = mask_head.loss(mask_pred, mask_targets, pos_labels)
                for name, value in loss_mask.items():
                    losses['s{}.{}'.format(
                        i, name)] = (value * lw if 'loss' in name else value)

            # refine bboxes
            if i < self.num_stages - 1:
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                roi_labels = bbox_targets[0]  # bbox_targets is a tuple
                with torch.no_grad():
                    proposal_list = bbox_head.refine_bboxes(
                        rois, roi_labels, bbox_pred, pos_is_gts, img_metas)

        return losses

    def simple_test(self, img, img_metas, proposals=None, rescale=False):

        x = self.extract_feat(img)
        # x = self.enhance_feature(x)
        edge_att, x = self.edge_enhance(x)

        if self.with_neck:
            x = self.neck(x)

        proposal_list = self.simple_test_rpn(
            x, img_metas,
            self.test_cfg.rpn) if proposals is None else proposals

        img_shape = img_metas[0]['img_shape']
        ori_shape = img_metas[0]['ori_shape']
        scale_factor = img_metas[0]['scale_factor']

        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_segm_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg.rcnn

        rois = bbox2roi(proposal_list)
        for i in range(self.num_stages):
            bbox_roi_extractor = self.bbox_roi_extractor[i]
            bbox_head = self.bbox_head[i]

            bbox_feats = bbox_roi_extractor(
                x[:len(bbox_roi_extractor.featmap_strides)], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)

            cls_score, bbox_pred,color_pred = bbox_head(bbox_feats)
            color_pred = torch.mean(color_pred, 1).reshape(-1, 1).expand(
                color_pred.size()[0],
                cls_score.size()[1])
            cls_score = cls_score * color_pred
            ms_scores.append(cls_score)

            if i < self.num_stages - 1:
                bbox_label = cls_score.argmax(dim=1)
                rois = bbox_head.regress_by_class(rois, bbox_label, bbox_pred,
                                                  img_metas[0])

        cls_score = sum(ms_scores) / self.num_stages
        det_bboxes, det_labels = self.bbox_head[-1].get_det_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.bbox_head[-1].num_classes)
        ms_bbox_result['ensemble'] = bbox_result

        if self.with_mask:
            if det_bboxes.shape[0] == 0:
                mask_classes = self.mask_head[-1].num_classes - 1
                segm_result = [[] for _ in range(mask_classes)]
            else:
                if isinstance(scale_factor, float):  # aspect ratio fixed
                    _bboxes = (
                        det_bboxes[:, :4] *
                        scale_factor if rescale else det_bboxes)
                else:
                    _bboxes = (
                        det_bboxes[:, :4] *
                        torch.from_numpy(scale_factor).to(det_bboxes.device)
                        if rescale else det_bboxes)

                mask_rois = bbox2roi([_bboxes])
                aug_masks = []
                for i in range(self.num_stages):
                    mask_roi_extractor = self.mask_roi_extractor[i]
                    mask_feats = mask_roi_extractor(
                        x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
                    if self.with_shared_head:
                        mask_feats = self.shared_head(mask_feats)
                    mask_pred = self.mask_head[i](mask_feats)
                    aug_masks.append(mask_pred.sigmoid().cpu().numpy())
                merged_masks = merge_aug_masks(aug_masks,
                                               [img_metas] * self.num_stages,
                                               self.test_cfg.rcnn)
                segm_result = self.mask_head[-1].get_seg_masks(
                    merged_masks, _bboxes, det_labels, rcnn_test_cfg,
                    ori_shape, scale_factor, rescale)
            ms_segm_result['ensemble'] = segm_result

        if self.with_mask:
            results = (ms_bbox_result['ensemble'], ms_segm_result['ensemble'])
        else:
            results = ms_bbox_result['ensemble']

        return results
