import copy
import glob
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

from data.utils import collate_fn_img_txt


class YePopDataset(Dataset):
    """
    Dataset for YePoP https://huggingface.co/datasets/Ejafa/ye-pop
    """

    def __init__(
        self,
        path: str,
        size: int = 1024,
        random_flip: bool = False,
        caption_type: str = "both",
    ):
        super().__init__()
        self.path = path
        self.size = size
        self.random_flip = random_flip
        self.caption_type = caption_type.lower()

        assert caption_type.lower() in [
            "llava",
            "cogvlm",
            "both",
        ], "Invalid Caption type, choose either 'llava', 'cogvlm', or 'both'"
        if self.caption_type != "both":
            self.caption_key = f"{self.caption_type}_caption"
        else:
            self.caption_key = None
        self.sample_threshold = 0.5

        self.samples = self.get_samples(self.path)

        transforms_list = [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(size=(self.size, self.size)),
            v2.Normalize(mean=[0.5], std=[0.5]),
        ]
        if self.random_flip:
            transforms_list.append(v2.RandomHorizontalFlip(p=0.5))
        self.transform = v2.Compose(transforms_list)

    def get_samples(self, path, valids_file=None):
        # load all entries from the jsons
        jsons = glob.glob(os.path.join(path, "json/*.json"))
        all_data = {}
        for json_file in jsons:
            with open(json_file) as f:
                d = json.load(f)
                all_data.update(d)
        all_data = dict(sorted(all_data.items()))

        # get all images
        image_paths = glob.glob(os.path.join(path, "images/**/*.jpg"), recursive=True)
        if len(image_paths) == 0:
            print("[YE-POP] Dataset Path seems incorrect - no images found")
        elif len(image_paths) < 450_000:
            print("[YE-POP] Dataset seems incomplete")
        # mappig from filename to full path
        image_dict = {
            str(int(img_path.split("/")[-1].replace(".jpg", ""))): img_path
            for img_path in image_paths
        }

        # add global paths to entry if image exists, else delete
        all_data_keys_copy = copy.deepcopy(list(all_data.keys()))
        for k in all_data_keys_copy:
            try:
                path_to_image = image_dict[k]
                all_data[k]["path_to_image"] = path_to_image
            except KeyError:
                # Image does not exist
                all_data.pop(k)

        if len(all_data) != len(image_dict):
            print("seems like there are images without metadata?")
        print(f"[YE-POP] Found {len(all_data.values())} entries and {len(image_dict)} image paths")
        return list(all_data.values())

    def __getitem__(self, index):

        sample = self.samples[index]

        image = Image.open(sample["path_to_image"])

        image = self.transform(image)

        if self.caption_key is None:
            caption_key = (
                "llava_caption" if np.random.rand() > self.sample_threshold else "cogvlm_caption"
            )
        else:
            caption_key = self.caption_key

        item_dict = {"pixel_values": image, "prompt": sample[caption_key]}
        return item_dict

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    instance_data_dir = "/ephemeral/datasets/ye-pop"
    random_flip = False
    resolution = 512
    caption_type = "both"
    take = 3

    train_dataset = YePopDataset(
        path=instance_data_dir,
        random_flip=random_flip,
        size=resolution,
        caption_type=caption_type,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn_img_txt,
        drop_last=True,
        prefetch_factor=2,
        pin_memory=True,
    )

    for i, batch in enumerate(train_dataloader):
        for k, v in batch.items():
            print(f"{i}th batch: {k}, {type(v)}")
        print(f"{i}th batch: {batch}")
        if i > take:
            break
