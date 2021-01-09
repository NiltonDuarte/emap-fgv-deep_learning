from PIL import Image
from os import walk

raw_folder = '../dataset/raw'
destiny_folder = '../dataset/preprocessed'


def process_NORMAL_images():
    raw_folder = f'{raw_folder}/NORMAL'
    destiny_folder = f'{destiny_folder}/NORMAL'
    _, _, images = next(walk(raw_folder))
    for image in images:
        image_path = f"{raw_folder}/{image}"
        destiny_path = f"{destiny_folder}/{image}"
        print(f'processing image {image}')
        im = Image.open(image_path)
        region = im.crop((100, 100, 1024, 1024))
        grayscale = region.convert('L')
        resized = grayscale.resize((256, 256), Image.LANCZOS)
        resized.save(destiny_path)


def process_COVID_images():
    raw_folder = f'{raw_folder}/COVID'
    destiny_folder = f'{destiny_folder}/COVID'
    _, _, images = next(walk(raw_folder))
    for image in images:
        image_path = f"{raw_folder}/{image}"
        destiny_path = f"{destiny_folder}/{image}"
        print(f'processing image {image}')
        im = Image.open(image_path)
        grayscale = im.convert('L')
        grayscale.save(destiny_path)


def upload_data():
    from google.cloud import storage
    bucket_name = 'emap-fgv'
    bucket_base_path = 'deep_learning/COVID/dataset/preprocessed'
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    for sub_folder in ['NORMAL', 'COVID']:
        path = f'{destiny_folder}/{sub_folder}'
        _, _, images = next(walk(path))
        for image in images:
            image_path = f"{path}/{image}"
            image_blob = bucket.blob(
                f'{bucket_base_path}/{sub_folder}/{image}')
            print(f"uploading {image_path}")
            image_blob.upload_from_filename(image_path)


if __name__ == '__main__':
    # process_NORMAL_images()
    # process_COVID_images()
    upload_data()
