from pathlib import Path
import numpy as np
import cv2
import torch
from isegm.data.base import ISDataset
from isegm.data.sample import DSample


class AdobeAlphamatteDataset(ISDataset):
    def __init__(self, dataset_path, split='train', **kwargs):
        super().__init__(**kwargs)
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.image_dir = self.dataset_path / 'original'
        self.mask_dir = self.dataset_path / 'mask'
        self.alpha_dir = self.dataset_path / 'alpha'

        # 디렉토리 존재 여부 확인
        for dir_path in [self.image_dir, self.mask_dir, self.alpha_dir]:
            if not dir_path.exists():
                raise FileNotFoundError(f"Directory not found: {dir_path}")
        
        self.dataset_samples = sorted([p.stem for p in self.image_dir.glob('*.jpg')])
        if len(self.dataset_samples) == 0:
            raise ValueError(f"No images found in {self.image_dir}")

    def get_sample(self, index) -> DSample:
        sample_id = self.dataset_samples[index]

        # 1. Load original image
        image = cv2.imread(str(self.image_dir / f'{sample_id}.jpg'))
        if image is None:
            raise FileNotFoundError(f"Failed to load image: {self.image_dir / f'{sample_id}.jpg'}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Load segmentation mask (already 0 or 255)
        seg_mask = cv2.imread(str(self.mask_dir / f'{sample_id}.png'), cv2.IMREAD_GRAYSCALE)
        if seg_mask.max() > 255 or seg_mask.min() < 0:
            raise ValueError(f"Invalid mask values in {sample_id}.png")
        seg_mask = (seg_mask > 127).astype(np.uint8)  # ensure binary 0/1
        seg_mask = seg_mask[:, :, None]               # [H, W, 1]

        # 3. Load alpha matte ground truth, normalize to [0, 1]
        alpha = cv2.imread(str(self.alpha_dir / f'{sample_id}.png'), cv2.IMREAD_GRAYSCALE)
        if alpha.max() > 255 or alpha.min() < 0:
            raise ValueError(f"Invalid alpha matte values in {sample_id}.png")
        alpha = alpha.astype(np.float32) / 255.0      # [H, W], float
        if alpha.max() > 1.0 or alpha.min() < 0.0:
            raise ValueError(f"Invalid normalized alpha matte values in {sample_id}.png")

        sample = DSample(
            image=image,
            encoded_masks=seg_mask,
            objects_ids=[1],
            sample_id=index,
            gt_mask=alpha
        )
        sample.gt_mask = alpha

        return sample

    def __getitem__(self, index):
        sample = self.get_sample(index)
        sample = self.augment_sample(sample)

        # 메모리 해제를 위한 명시적 삭제
        result = {
            'images': self.to_tensor(sample.image),
            'seg_mask': self.to_tensor(sample._encoded_masks).float(),
            'instances': self.to_tensor(sample.gt_mask).float()
        }
        del sample
        return result
