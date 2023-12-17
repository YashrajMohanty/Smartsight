import numpy as np
import torchvision.transforms.v2 as v2

def apply_min_size(sample, size, image_interpolation_method=v2.InterpolationMode.BICUBIC):
    """Rezise the sample to ensure the given size. Keeps aspect ratio.

    Args:
        sample (tensor): sample
        size (tuple): image size

    Returns:
        tuple: new size
    """
    shape = sample["disparity"].shape

    if shape[0] >= size[0] and shape[1] >= size[1]:
        return sample

    scale = [0, 0]
    scale[0] = size[0] / shape[0]
    scale[1] = size[1] / shape[1]

    scale = max(scale)

    shape[0] = np.ceil(scale * shape[0])
    shape[1] = np.ceil(scale * shape[1])

    # resize

    sample["image"] = v2.Resize(size=shape, interpolation=image_interpolation_method)(sample["image"])
    #sample["image"] = cv2.resize(
    #    sample["image"], tuple(shape[::-1]), interpolation=image_interpolation_method
    #)

    sample["disparity"] = v2.Resize(size=shape, interpolation=v2.InterpolationMode.NEAREST)(sample["disparity"])
    #sample["disparity"] = cv2.resize(
    #    sample["disparity"], tuple(shape[::-1]), interpolation=cv2.INTER_NEAREST
    #)

    sample["mask"] = v2.Resize(size=shape, interpolation=v2.InterpolationMode.NEAREST)(sample["mask"])
    sample["mask"] = sample["mask"].bool()
    #sample["mask"] = cv2.resize(
    #    sample["mask"].astype(np.float32),
    #    tuple(shape[::-1]),
    #    interpolation=cv2.INTER_NEAREST,
    #)
    #sample["mask"] = sample["mask"].astype(bool)

    return shape


class Resize(object):
    """Resize sample to given size (width, height).
    """

    def __init__(
        self,
        width,
        height,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
        image_interpolation_method=v2.InterpolationMode.BICUBIC,
    ):
        """Init.

        Args:
            width (int): desired output width
            height (int): desired output height
            resize_target (bool, optional):
                True: Resize the full sample (image, mask, target).
                False: Resize image only.
                Defaults to True.
            keep_aspect_ratio (bool, optional):
                True: Keep the aspect ratio of the input sample.
                Output sample might not have the given width and height, and
                resize behaviour depends on the parameter 'resize_method'.
                Defaults to False.
            ensure_multiple_of (int, optional):
                Output width and height is constrained to be multiple of this parameter.
                Defaults to 1.
            resize_method (str, optional):
                "lower_bound": Output will be at least as large as the given size.
                "upper_bound": Output will be at max as large as the given size. (Output size might be smaller than given size.)
                "minimal": Scale as least as possible.  (Output size might be smaller than given size.)
                Defaults to "lower_bound".
        """
        self.__width = width
        self.__height = height

        self.__resize_target = resize_target
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        #y = torch.mul(torch.div(x, self.__multiple_of, rounding_mode="trunc"), self.__multiple_of)
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if max_val is not None and y > max_val:
            #y = torch.mul(torch.div(x, self.__multiple_of, rounding_mode="floor"), self.__multiple_of)
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if y < min_val:
            #y = torch.mul(torch.div(x, self.__multiple_of, rounding_mode="trunc"), self.__multiple_of)
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        # determine new height and width
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(
                    f"resize_method {self.__resize_method} not implemented"
                )

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, min_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, min_val=self.__width
            )
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, max_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, max_val=self.__width
            )
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def __call__(self, sample):
        width, height = self.get_size(sample["image"].shape[1], sample["image"].shape[0])

        # resize sample
        sample["image"] = v2.Resize(size=(height, width), interpolation=v2.InterpolationMode.BICUBIC, antialias=False)(sample["image"])
        #sample["image"] = cv2.resize(sample["image"], (width, height), interpolation=self.__image_interpolation_method)

        if self.__resize_target:
            if "disparity" in sample:
                sample["disparity"] = v2.Resize(size=(height, width), interpolation=v2.InterpolationMode.NEAREST)(sample["disparity"])
                #sample["disparity"] = cv2.resize(sample["disparity"], (width, height), interpolation=cv2.INTER_NEAREST)

            if "depth" in sample:
                sample["depth"] = v2.Resize(size=(height, width), interpolation=v2.InterpolationMode.NEAREST)(sample["depth"])
                #sample["depth"] = cv2.resize(sample["depth"], (width, height), interpolation=cv2.INTER_NEAREST)

            sample["mask"] = v2.Resize(size=(height, width), interpolation=v2.InterpolationMode.NEAREST)(sample["mask"])
            sample["mask"] = sample["mask"].bool()
            #sample["mask"] = cv2.resize(sample["mask"].astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
            #sample["mask"] = sample["mask"].astype(bool)

        return sample


class NormalizeImage(object):
    """Normlize image by given mean and std.
    """

    def __init__(self, mean, std):
        self.__mean = mean
        self.__std = std

    def __call__(self, sample):
        v2.functional.normalize(sample["image"], self.__mean, self.__std, True)
        #sample["image"] = torch.div(torch.sub(sample["image"], self.__mean), self.__std)
        #sample["image"] = (sample["image"] - self.__mean) / self.__std

        return sample


class PrepareForNet(object):
    """Prepare sample for usage as network input.
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        """
        image = np.transpose(sample["image"], (2, 0, 1))
        sample["image"] = np.ascontiguousarray(image).astype(np.float32)

        if "mask" in sample:
            sample["mask"] = sample["mask"].astype(np.float32)
            sample["mask"] = np.ascontiguousarray(sample["mask"])

        if "disparity" in sample:
            
            disparity = sample["disparity"].astype(np.float32)
            sample["disparity"] = np.ascontiguousarray(disparity)

        if "depth" in sample:
            depth = sample["depth"].astype(np.float32)
            sample["depth"] = np.ascontiguousarray(depth)
        """

        return sample
    

not_small_transform = v2.Compose(
    [
        lambda img: {"image": img / 255.0},
        Resize(
            256,
            256,
            resize_target=None,
            keep_aspect_ratio=True,
            ensure_multiple_of=32,
            resize_method="upper_bound",
            image_interpolation_method=v2.InterpolationMode.BICUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #PrepareForNet(),
        lambda sample: (sample["image"]).unsqueeze(0),
    ]
)

small_transform = v2.Compose(
    [
        lambda img: {"image": img / 255.0},
        v2.Resize([256,256], interpolation=v2.InterpolationMode.BICUBIC, antialias=False),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        lambda sample: (sample["image"]).unsqueeze(0),
    ]
)