import json
import numpy as np
from PIL import Image

_M_RGB2YUV = [[0.299, 0.587, 0.114], [-0.14713, -0.28886, 0.436], [0.615, -0.51499, -0.10001]]

def convert_PIL_to_numpy(image, format):
        """
        Convert PIL image to numpy array of target format.

        Args:
            image (PIL.Image): a PIL image
            format (str): the format of output image

        Returns:
            (np.ndarray): also see `read_image`
        """
        if format is not None:
            # PIL only supports RGB, so convert to RGB and flip channels over below
            conversion_format = format
            if format in ["BGR", "YUV-BT.601"]:
                conversion_format = "RGB"
            image = image.convert(conversion_format)
        image = np.asarray(image)
        # PIL squeezes out the channel dimension for "L", so make it HWC
        if format == "L":
            image = np.expand_dims(image, -1)

        # handle formats not supported by PIL
        elif format == "BGR":
            # flip channels if needed
            image = image[:, :, ::-1]
        elif format == "YUV-BT.601":
            image = image / 255.0
            image = np.dot(image, np.array(_M_RGB2YUV).T)

        return image

def rgb2id(color):
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])



def main(input_image,seg_image_root):
    dataset= json.load('./sub_ris.json','r')
    mask=[]
    data=dataset[input_image]
    seg_img=data['pan_seg_file']
    id=data['id']
    seg_img=Image.open(seg_image_root + seg_img)

    seg_map = convert_PIL_to_numpy(seg_img, "RGB")
    seg_map= rgb2id(seg_map)
    mask.append((seg_map == id).tolist())
    
    return mask

input_image= 'number(0-999)'
seg_image_root='path to panoptic_2017'
mask=main(input_image,seg_image_root)