import os
import urllib.request
import zipfile
from tqdm import tqdm

def download_flickr8k():
    # URLs for the dataset
    image_url = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"
    text_url = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"
    
    # Create a directory for the dataset
    os.makedirs("flickr8k", exist_ok=True)

    # Download and extract images
    print("Downloading Flickr8k images...")
    urllib.request.urlretrieve(image_url, "flickr8k/images.zip")
    with zipfile.ZipFile("flickr8k/images.zip", 'r') as zip_ref:
        zip_ref.extractall("flickr8k")
    os.remove("flickr8k/images.zip")

    # Download and extract text
    print("Downloading Flickr8k text data...")
    urllib.request.urlretrieve(text_url, "flickr8k/text.zip")
    with zipfile.ZipFile("flickr8k/text.zip", 'r') as zip_ref:
        zip_ref.extractall("flickr8k")
    os.remove("flickr8k/text.zip")

    print("Flickr8k dataset downloaded and extracted.")

def load_descriptions(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    
    descriptions = {}
    for line in text.split('\n'):
        tokens = line.split()
        if len(line) < 2:
            continue
        image_id, image_desc = tokens[0], tokens[1:]
        image_id = image_id.split('.')[0]
        image_desc = ' '.join(image_desc)
        if image_id not in descriptions:
            descriptions[image_id] = []
        descriptions[image_id].append(image_desc)
    return descriptions

def main():
    download_flickr8k()
    
    # Load image descriptions
    filename = 'flickr8k/Flickr8k.token.txt'
    descriptions = load_descriptions(filename)
    
    print(f'Loaded: {len(descriptions)} image descriptions')
    
    # Print a sample
    image_id = list(descriptions.keys())[0]
    print(f'\nSample image ID: {image_id}')
    print(f'Sample captions:')
    for caption in descriptions[image_id]:
        print(f' - {caption}')

if __name__ == '__main__':
    main()