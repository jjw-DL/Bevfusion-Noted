from mmcv.runner import EpochBasedRunner
from mmcv.runner.builder import RUNNERS

@RUNNERS.register_module()
class CustomEpochBasedRunner(EpochBasedRunner):
    def set_dataset(self, dataset):
        self._dataset = dataset


    def train(self, data_loader, **kwargs):
        # update the schedule for data augmentation
        for dataset in self._dataset: # CBGSDataset
            dataset.set_epoch(self.epoch) # 0
        super().train(data_loader, **kwargs)
