from dataclasses import dataclass
from typing import Protocol, TypeVar

import matplotlib.pyplot as plt
import numpy as np

from numpy.typing import NDArray

from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

###############################################################################
### Type Definitions ##########################################################
###############################################################################

ResizeType = tuple[int | None, int | None]

T = TypeVar("T", covariant=True)
U = TypeVar("U", bound=np.generic, contravariant=True)
V = TypeVar("V", bound=np.generic, covariant=True)


@dataclass
class QuantizedColourImage:
    """Dataclass for quantized colour image."""

    image_data: NDArray[np.int_]
    centers: NDArray[np.float32]


###############################################################################
### Protocols #################################################################
###############################################################################


class ImageTemplate(Protocol[T, U]):
    def __call__(self, x: NDArray[U]) -> T: ...


class CharFormater(Protocol[V]):
    def __call__(self, i: int, centers: NDArray[np.float32]) -> V: ...


###############################################################################
### Image Template Functions ##################################################
###############################################################################


def str_template(x: NDArray[np.str_]) -> np.str_:
    """String template function."""
    final_str = ""
    for row in x:
        final_str += "".join(row) + "\n"
    return np.str_(final_str[:-1])


def html_str_template(x: NDArray[np.str_]) -> np.str_:
    """HTML string template function."""
    str_vals = str_template(x)
    final_html = f"""
    <!DOCTYPE html>
    <html>
    <body>
        <pre style="font-family: monospace; font-size: 14px; line-height: 1ch;">{str_vals}</pre>
    </body>
    </html>
    """
    return np.str_(final_html)


###############################################################################
### Image Manipulation Class ##################################################
###############################################################################


class ImageManipulate:
    """Manipulate images."""

    def __init__(self, path_to_image: str) -> None:
        """
        Parameters
        ----------
        path_to_image : str
            Path to the image to be converted.
        """

        self.path_to_image = path_to_image
        self._load_img_data()

    def _load_img_data(self) -> None:
        """Load image data from path."""

        image = Image.open(self.path_to_image)
        self.image = image.convert("RGB")

    def resize(self, resize: ResizeType | None = None) -> None:
        """Resize image.

        Parameters
        ----------
        resize : (int, int)
            Resizing the image. Each pixel will be an ascii character. None or
            (None, None) retains default size. A number and None will retain the
            original ratio.
        """
        if resize is None:
            return

        # resize
        hresize, vresize = resize
        horig, vorig = self.image.size
        if hresize is None and vresize is not None:
            scale = vresize / vorig
            hresize = int(horig * scale)
            resize = (hresize, vresize)

        elif vresize is None and hresize is not None:
            scale = hresize / horig
            vresize = int(vorig * scale)
            resize = (hresize, vresize)

        elif hresize is not None and vresize is not None:
            resize = (hresize, vresize)

        else:
            return

        self.image = self.image.resize(resize)

    def get_colour_quantize_image(
        self, components: int, seed: int | None
    ) -> QuantizedColourImage:
        """Quantize the image colour with k means clustering.

        Parameters
        ----------
        components : int
            Number of components/characters to convert image into.
        seed : int
            The random seed to use for KMeans clustering. Can be None.

        Returns
        -------
        quantized_image : QuantizedColourImage
            The quantized colour image.
        """

        # open and reshape in matplotlib
        image_data = np.array(self.image, dtype=float)

        vpx, hpx, rgb = image_data.shape
        image_data = image_data.reshape(vpx * hpx, rgb)

        # normalise
        scaler = StandardScaler().fit(image_data)
        img_scale = scaler.transform(image_data)

        # fit and apply k means clustering
        kmeans = KMeans(n_clusters=components, random_state=seed)
        kmeans.fit(img_scale)
        # this seems to keep ordering correct as opposed to predict
        # important for cartoon filter, not so much ascii art
        image_quantized = kmeans.labels_
        image_quantized = image_quantized.reshape(vpx, hpx)

        return QuantizedColourImage(
            image_data=image_quantized,
            centers=scaler.inverse_transform(kmeans.cluster_centers_),
        )


###############################################################################
### Image Manipulation Functions ##############################################
###############################################################################


def img_to_quantized_output(
    path_to_image: str,
    image_template: ImageTemplate[T, U],
    pixel_formater: CharFormater[U],
    components: int = 20,
    resize: ResizeType | None = None,
    seed: int | None = None,
) -> T:
    """Convert image to quantized string representation.

    Parameters
    ----------
    path_to_image : str
        Path to the image to be converted.
    str_template : function
        Function to format the final string. If None, default function is used.
    char_formater : function
        Function to format each character. If None, default function is used.
    components : int
        Number of components/characters to convert image into.
    resize : (int, int)
        Resizing the image. Each pixel will be an ascii character. None or
        (None, None) retains default size. A number and None will retain the
        original ratio.
    seed : int
        The random seed to use for KMeans clustering. Can be None.

    Returns
    -------
    quantized_str : str
        The image converted to quantized string representation.
    """

    img_obj = ImageManipulate(path_to_image=path_to_image)
    img_obj.resize(resize=resize)
    quantized_color_image = img_obj.get_colour_quantize_image(components, seed=seed)

    # convert to quantized string

    quantized_output = []
    for row in quantized_color_image.image_data:
        output_row = []
        for v in row:
            pixel = pixel_formater(v, quantized_color_image.centers)
            output_row.append(pixel)
        quantized_output.append(output_row)
    quantized_array = np.array(quantized_output)

    return image_template(quantized_array)


def img_to_ascii(
    path_to_image: str,
    components: int = 20,
    resize: ResizeType | None = None,
    ascii_str: str = "-:`!@#$%^&*0123456789qwertyuiopasdfghjklzxcvbnm",
    colour: bool = True,
    seed: int | None = None,
):
    """Convert image to ascii art.

    Parameters
    ----------
    path_to_image : str
        Path to the image to be converted.
    components : int
        Number of components/characters to convert image into.
    resize : (int, int)
        Resizing the image. Each pixel will be an ascii character. None or
        (None, None) retains default size. A number and None will retain the
        original ratio.
    colour : bool
        Whether to output ascii art in colour.
    ascii_str : str
        The ascii characters to use. If string is longer than components, rest
        of string is ignored.

    Returns
    -------
    ascii_art : str
        The image converted to ascii art.
    """
    if components > len(ascii_str):
        raise ValueError(
            "ascii_str is not long enough for the number of components specified."
        )

    def ascii_char_formater(i: int, centers: np.ndarray) -> np.str_:
        char = ascii_str[i]
        if colour:
            r, g, b = centers[i].astype(int)
            char = f"\033[38;2;{r};{g};{b}m" + char + "\033[0m"
        return np.str_(char)

    return img_to_quantized_output(
        path_to_image=path_to_image,
        image_template=str_template,
        pixel_formater=ascii_char_formater,
        components=components,
        resize=resize,
        seed=seed,
    )


def img_to_html_ascii(
    path_to_image: str,
    path_to_html: str,
    components: int = 20,
    resize: ResizeType | None = None,
    ascii_str: str = "-:`!@#$%^&*0123456789qwertyuiopasdfghjklzxcvbnm",
    colour: bool = True,
    seed: int | None = None,
) -> None:
    """Convert image to ascii art in html format.

    Parameters
    ----------
    path_to_image : str
        Path to the image to be converted.
    path_to_html : str
        Path to output html file.
    components : int
        Number of components/characters to convert image into.
    resize : (int, int)
        Resizing the image. Each pixel will be an ascii character. None or
        (None, None) retains default size. A number and None will retain the
        original ratio.
    ascii_str : str
        The ascii characters to use. If string is longer than components, rest
        of string is ignored.
    """
    if components > len(ascii_str):
        raise ValueError(
            "ascii_str is not long enough for the number of components specified."
        )

    def html_ascii_char_formater(i: int, centers: np.ndarray) -> np.str_:
        char = ascii_str[i]
        if colour:
            r, g, b = centers[i].astype(int)
            hex_str = f"#{r:02x}{g:02x}{b:02x}"
            char = f'<span style="color:{hex_str}">{char}</span>'
        return np.str_(char)

    html_str = img_to_quantized_output(
        path_to_image=path_to_image,
        image_template=html_str_template,
        pixel_formater=html_ascii_char_formater,
        components=components,
        resize=resize,
        seed=seed,
    )

    # Export to html file
    if not path_to_html.endswith(".html"):
        path_to_html += ".html"

    with open(path_to_html, "w") as f:
        f.write(html_str)


def cartoon_filter(
    path_to_image: str,
    path_to_output: str,
    components: int = 20,
    resize: ResizeType | None = None,
    seed: int | None = None,
) -> None:
    """Filter the image through a cartoon effect.

    Parameters
    ----------
    path_to_image : str
        Path to the image to be converted.
    path_to_output : str
        Path to output image file.
    components : int
        Number of components/characters to convert image into.
    resize : (int, int)
        Resizing the image. Each pixel will be an ascii character. None or
        (None, None) retains default size. A number and None will retain the
        original ratio.
    seed : int
        The random seed to use for KMeans clustering. Can be None.

    """

    def cartoon_char_formater(i: int, centers: np.ndarray) -> np.float32:
        return centers[i] / 255.0

    quantized_colour_image = img_to_quantized_output(
        path_to_image=path_to_image,
        image_template=lambda x: x,
        pixel_formater=cartoon_char_formater,
        components=components,
        resize=resize,
        seed=seed,
    )

    # save image
    plt.imsave(path_to_output, quantized_colour_image)


def img_to_html_pixelart(
    path_to_image: str,
    path_to_html: str,
    components: int = 20,
    resize: ResizeType | None = None,
    seed: int | None = None,
) -> None:
    """Convert image to pixel art.

    Parameters
    ----------
    path_to_image : str
        Path to the image to be converted.
    components : int
        Number of components/characters to convert image into.
    resize : (int, int)
        Resizing the image. Each pixel will be an ascii character. None or
        (None, None) retains default size. A number and None will retain the
        original ratio.
    seed : int
        The random seed to use for KMeans clustering. Can be None.
    """

    def pixel_char_formater(i: int, centers: np.ndarray) -> np.str_:
        r, g, b = centers[i].astype(int)
        hex_str = f"#{r:02x}{g:02x}{b:02x}"
        return np.str_(f'<span style="color:{hex_str}">█</span>')

    pixel_output = img_to_quantized_output(
        path_to_image=path_to_image,
        image_template=html_str_template,
        pixel_formater=pixel_char_formater,
        components=components,
        resize=resize,
        seed=seed,
    )

    # Export to html file
    if not path_to_html.endswith(".html"):
        path_to_html += ".html"

    with open(path_to_html, "w") as f:
        f.write(pixel_output)
