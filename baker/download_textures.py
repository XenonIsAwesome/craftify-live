from PIL import Image
from tqdm import tqdm
import requests
import sys
import zipfile
import os
from baker.objects.obj_cache import ObjectCache


def get_json_file(url):
    with requests.get(url) as response:
        return response.json()


objc = ObjectCache.get_instance()
objc.version_manifest_v2_url = "https://piston-meta.mojang.com/mc/game/version_manifest_v2.json"
objc.version_manifest_v2 = get_json_file(objc.version_manifest_v2_url)


def get_version_tag():
    version_tag = sys.argv[1]
    if version_tag == "latest":
        version_type = sys.argv[2]
        version_tag = objc.version_manifest_v2["latest"][version_type]
    
    objc.version_tag = version_tag


def get_game_version_manifest():
    for version in objc.version_manifest_v2["versions"]:
        if version["id"] == objc.version_tag:
            objc.game_version_manifest_url = version["url"]
            objc.game_version_manifest = get_json_file(objc.game_version_manifest_url)
            break
    else:
        raise KeyError(objc.version_tag)


def download_version_jar():
    objc.client_jar_url = objc.game_version_manifest["downloads"]["client"]["url"]
    client_jar = requests.get(objc.client_jar_url)
    with open(f"client.jar", "wb") as f:
        f.write(client_jar.content)


def unzip_assets():
    # unzip assets/minecraft/textures/block/*.png
    with zipfile.ZipFile(f"client.jar", "r") as zip_ref:
        objc.filtered_filelist = [f for f in zip_ref.filelist if f.filename.startswith("assets/minecraft/textures/block/") and f.filename.endswith(".png")]        
        for f in tqdm(objc.filtered_filelist, desc="Extracting textures"):
            zip_ref.extract(f)

    os.remove("client.jar")


def avg_alpha_filter(f, img):
    sum_alpha = 0

    for y in range(img.height):
        for x in range(img.width):
            rgba = img.getpixel((x, y))
            sum_alpha += rgba[3]

    total_pixels = img.width * img.height
    avg_alpha = sum_alpha // total_pixels

    return avg_alpha < 255


def filter_textures():
    filters = [
        lambda f, img: "debug" in f.filename,
        lambda f, img: img.size != (16, 16),
        avg_alpha_filter
    ]

    objc.unfiltered_count = 0
    for f in tqdm(objc.filtered_filelist, desc="Filtering textures"):
        img = Image.open(f.filename)
        img = img.convert("RGBA")

        for flt in filters:
            if flt(f, img):
                try:
                    os.remove(f.filename)
                    break
                except:
                    pass
        else:
            objc.unfiltered_count = objc.unfiltered_count + 1


def download_textures():
    get_version_tag()
    print(f"Downloading textures for version {objc.version_tag}")
    get_game_version_manifest()
    download_version_jar()
    unzip_assets()
    filter_textures()
    print("Done.", objc.unfiltered_count, "textures left after filtering.")