import os
from PIL import Image
from struct import pack
from tqdm import tqdm

def get_score(img_path):
    with Image.open(img_path) as img:
        img = img.convert("RGBA")

        sum_r = 0
        sum_g = 0
        sum_b = 0

        for y in range(img.height):
            for x in range(img.width):
                rgba = img.getpixel((x, y))
                sum_r += rgba[0]
                sum_g += rgba[1]
                sum_b += rgba[2]

        total_pixels = img.width * img.height
        avg_r = sum_r // total_pixels
        avg_g = sum_g // total_pixels
        avg_b = sum_b // total_pixels

        return avg_r + avg_g + avg_b


def pack_textures():
    # pack all the pngs in assets/minecraft/textures/block/ into a texture atlas 512x288 pixels
    # background should be #000000
    BASE_PATH = 'assets/minecraft/textures/block/'
    images = sorted(os.listdir(BASE_PATH), key=lambda x: get_score(BASE_PATH + x))

    # 16x16 images
    img_width = 16
    img_height = 16
    img_per_row = 32
    img_per_col = 18

    texture = Image.new('RGBA', (img_width * img_per_row, img_height * img_per_col), (0, 0, 0, 255))

    scores = []
    for i, img_path in tqdm(enumerate(images), desc="Packing textures", total=len(images)):
        img = Image.open(BASE_PATH + img_path)
        img = img.convert("RGBA")

        x = i % img_per_row
        y = i // img_per_row

        if x == 0:
            score = get_score(BASE_PATH + img_path)
            scores.append(score)



        texture.paste(img, (x * img_width, y * img_height))

    texture.save('assets/atlas.png')

    print("Scores: ", scores)
    fmt = f'{len(scores)}Q'
    data = pack(fmt, *scores)

    with open('assets/scores.bin', 'wb') as f:
        f.write(data)
