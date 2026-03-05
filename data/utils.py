import torch


def collate_fn_img_txt(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    prompt = [example["prompt"] for example in examples]

    return {
        "pixel_values": pixel_values,
        "prompt": prompt,
    }
