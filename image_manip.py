import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class ImageManip:
    """Manipulate images."""

    def __init__(self, path_to_image, resize=(None, None)):
        """
        Parameters
        ----------
        path_to_image : str
            Path to the image to be converted.
        resize : (int, int)
            Resizing the image. Each pixel will be an ascii character. (None, None)
            retains default size. A number and None will retain the original ratio.
        """

        self.path_to_image = path_to_image
        self.resize = resize

    def preprocess_img(self):
        """Resize image."""

        # resize
        image = Image.open(self.path_to_image)
        image = image.convert("RGB")

        horig, vorig = image.size
        if self.resize[0] is not None and self.resize[1] is None:
            scale = self.resize[0] / horig
            self.resize = (self.resize[0], int(vorig * scale))
        elif self.resize[0] is None and self.resize[1] is not None:
            scale = self.resize[1] / vorig
            self.resize = (int(horig * scale), self.resize[1])
        if self.resize[0] is not None and self.resize[1] is not None:
            new_image = image.resize(self.resize)
        else:
            new_image = image

        new_image_save_path = (
            "_temp_" + os.path.splitext(self.path_to_image)[0] + ".png"
        )
        new_image.save(new_image_save_path)

    def fit(self, components, seed):
        """Fit the image with k means clustering.

        Parameters
        ----------
        components : int
            Number of components/characters to convert image into.
        seed : int
            The random seed to use for KMeans clustering. Can be None.

        Returns
        -------
        data : dict
            img (2darray of ints) is the fitted image, each value is a component.
            img_orig (2darray of floats) is the original img.
            centers (2darray of floats) is the center of the clusters.
        """

        # open and reshape in matplotlib
        temp_path = "_temp_" + os.path.splitext(self.path_to_image)[0] + ".png"
        img = mpimg.imread(temp_path)
        img = 256 * img.astype(float)

        os.remove(temp_path)  # remove temp image
        vpx, hpx, rgb = img.shape
        img = img.reshape(vpx * hpx, rgb)

        # normalise
        scaler = StandardScaler().fit(img)
        img_scale = scaler.transform(img)

        # fit and apply k means clustering
        kmeans = KMeans(n_clusters=components, random_state=seed)
        kmeans.fit(img_scale)
        # this seems to keep ordering correct as opposed to predict
        # important for cartoon filter, not so much ascii art
        img_trans = kmeans.labels_
        img_trans = img_trans.reshape(vpx, hpx)

        return {
            "img": img_trans,
            "img_orig": img.reshape(vpx, hpx, rgb),
            "centers": scaler.inverse_transform(kmeans.cluster_centers_),
        }


def rgb_str(str, r, g, b):
    """Return string in rgb colour for terminal output."""
    return f"\033[38;2;{r};{g};{b}m" + str + "\033[0m"


def img_to_ascii(
    path_to_image,
    components=20,
    resize=(None, None),
    ascii_str="-:`!@#$%^&*0123456789qwertyuiopasdfghjklzxcvbnm",
    colour=True,
    seed=None,
):
    """Convert image to ascii art.

    Parameters
    ----------
    path_to_image : str
        Path to the image to be converted.
    components : int
        Number of components/characters to convert image into.
    resize : (int, int)
        Resizing the image. Each pixel will be an ascii character. (None, None)
        retains default size. A number and None will retain the original ratio.
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

    img_obj = ImageManip(path_to_image=path_to_image, resize=resize)
    img_obj.preprocess_img()
    data = img_obj.fit(components, seed=seed)

    img = data["img"]
    centers = data["centers"]

    # convert to ascii art
    ascii_art = ""
    for row in img:
        ascii_row = ""
        for i in row:
            char = ascii_str[i]
            if colour:
                r, g, b, *_ = centers[i].astype(int)
                char = rgb_str(char, r, g, b)

            ascii_row = ascii_row + char
        ascii_art = ascii_art + ascii_row + "\n"

    return ascii_art[:-1]  # remove last newline


def rgb_to_hex(r, g, b):
    """Convert rgb values to hex string."""
    return "#{:02x}{:02x}{:02x}".format(r, g, b)


def img_to_html_ascii(
    path_to_image,
    out_html,
    components=20,
    resize=(None, None),
    ascii_str="-:`!@#$%^&*0123456789qwertyuiopasdfghjklzxcvbnm",
    seed=None,
):
    """Convert image to ascii art in html format.

    Parameters
    ----------
    path_to_image : str
        Path to the image to be converted.
    components : int
        Number of components/characters to convert image into.
    resize : (int, int)
        Resizing the image. Each pixel will be an ascii character. (None, None)
        retains default size. A number and None will retain the original ratio.
    ascii_str : str
        The ascii characters to use. If string is longer than components, rest
        of string is ignored.

    Returns
    -------
    ascii_html_art : str
        The image converted to ascii art in html format.
    """

    if components > len(ascii_str):
        raise ValueError(
            "ascii_str is not long enough for the number of components specified."
        )

    img_obj = ImageManip(path_to_image=path_to_image, resize=resize)
    img_obj.preprocess_img()
    data = img_obj.fit(components, seed=seed)

    img = data["img"]
    centers = data["centers"]
    hex_centers = [rgb_to_hex(*c.astype(int)) for c in centers]

    ascii_art = ""
    for row in img:
        ascii_row = ""
        for i in row:
            char = ascii_str[i]
            char = f'<span style="color:{hex_centers[i]}">{char}</span>'

            ascii_row = ascii_row + char
        ascii_art = ascii_art + ascii_row + "\n"

    # HTML output
    name = os.path.splitext(os.path.basename(path_to_image))[0]
    final_html = final_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{name}</title>
    </head>
    <body>
        <pre style="font-family: monospace; font-size: 14px; line-height: 1;">{ascii_art}</pre>
    </body>
    </html>
    """

    # Export to html file
    if not out_html.endswith(".html"):
        out_html = out_html + ".html"

    with open(f"{out_html}", "w") as f:
        f.write(final_html)


def cartoon_filter(
    path_to_image,
    out_image,
    components=10,
    resize=(None, None),
    colours=None,
    seed=None,
):
    """Filter the image through a cartoon effect."""

    if colours is not None:
        if len(colours) != len(components):
            raise ValueError("Given colours are not the same length as components.")

    img_obj = ImageManip(path_to_image=path_to_image, resize=resize)
    img_obj.preprocess_img()
    data = img_obj.fit(components, seed=seed)

    # place all the rgb colours into the correct places
    img = data["img"]
    centers = data["centers"]
    filt_img = []
    for row in img:
        filt_img_row = []
        for pix in row:
            filt_img_row.append(centers[pix])
        filt_img.append(filt_img_row)
    filt_img = np.array(filt_img)

    # save image
    plt.imsave(out_image, filt_img)
