from tqdm import tqdm
import cv2
from PIL import Image

def convert_dataset(df, transform):
    data = []
    for i in tqdm(range(df.shape[0])):
        image = cv2.imread(df.iloc[i]['wsi_image'])
        image = transform(image = image)['image']
        image = Image.fromarray(image)

        q = df.iloc[i]['q']
        a = df.iloc[i]['a']

        data.append(
            {'messages': [
                {'role': 'user', 'content': [{'type': 'text','text': f'{q}'}, {'type': 'image', 'image': image}]},
                {'role': 'assistant', 'content': [{'type': 'text', 'text': f'{a}'}]}]}
        )
    
    return data