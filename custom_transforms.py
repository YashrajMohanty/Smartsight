import torchvision.transforms.v2 as v2

small_transform = v2.Compose(
    [
        lambda img: {"image": img / 255.0},
        v2.Resize([256,256], interpolation=v2.InterpolationMode.BICUBIC, antialias=False),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        lambda sample: (sample["image"]).unsqueeze(0),
    ])