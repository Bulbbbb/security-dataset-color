from .registry import DATASETS
from .coco import CocoDataset

@DATASETS.register_module
class PIDray(CocoDataset):

    def __init__(self, edge_map_prefix, **kwargs):
        super().__init__(**kwargs)
        self.edge_map_prefix = edge_map_prefix

    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['edge_map_prefix'] = self.edge_map_prefix
