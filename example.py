import image_manip

# print(img_to_ascii('fringes_cutout.png', resize=(150, None)))

# cartoon_filter('screenshot.png', 'filtered_ss.png', resize=(800, None))

image_manip.img_to_html_ascii(
    "screenshot.png",
    "screenshot.html",
    resize=(150, None),
)

image_manip.img_to_html_ascii(
    "coco_1.jpg",
    "coco_1.html",
    resize=(150, None),
)


image_manip.img_to_html_ascii(
    "fringes_cutout.png",
    "fringes_cutout.html",
    resize=(150, None),
)
