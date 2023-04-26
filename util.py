"""Utilities
"""
import re
import base64

import numpy as np

from PIL import Image
from io import BytesIO


def upload_image(encoded_string):
    decoded_bytes = base64.b64decode(encoded_string)
    # Create an image object from the decoded bytes
    img = Image.open(BytesIO(decoded_bytes))
    print( 'Image uploaded successfully')
    return img

    # Do something with the image, such as save it to disk or process it



def base64_to_pil(img_base64):
    """
    Convert base64 image data to PIL image
    """
    image_data = base64.b64decode(img_base64) #re.sub('^data:image/.+;base64,', '', img_base64)
    pil_image = Image.open(BytesIO(image_data))
    return pil_image


def np_to_base64(img_np):
    """
    Convert numpy image (RGB) to base64 string
    """
    img = Image.fromarray(img_np.astype('uint8'), 'RGB')
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return u"data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("ascii")
