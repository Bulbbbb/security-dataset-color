from mmdet.datasets import DATASETS
from matplotlib import pyplot as plt


def test_pidray():
    data_root = '/mnt/d/myWorkspace/datasets/PIDray/pidray/'
    img_norm_cfg = dict(mean=[127.5, 127.5, 127.5],
                        std=[127.5, 127.5, 127.5],
                        to_rgb=True)
    pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadEdgeMapFromFile'),
        # dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
        dict(type='BGResize', img_scale=(500, 500), keep_ratio=True),
        dict(type='BGRandomFlip', flip_ratio=0.5),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='BGPad', size_divisor=32),
        # dict(type='BGFormatBundle'),
        # dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'edge_map']),
    ]
    dataconfig = {
        'ann_file': data_root + 'annotations/xray_test.json',
        'img_prefix': data_root + 'train/',
        'edge_map_prefix': data_root + 'train_edge_mask/',
        'pipeline': pipeline
    }
    dataset = DATASETS.get('PIDray')
    pidray = dataset(**dataconfig)

    index = 0
    meta = pidray[index]
    img, edge_map = meta['img'], meta['edge_map']
    plt.imshow(img)
    plt.show()
    plt.imshow(edge_map)
    plt.show()


if __name__ == '__main__':
    test_pidray()
