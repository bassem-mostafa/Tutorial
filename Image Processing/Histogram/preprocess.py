import os
import cv2
from pathlib import Path
from tqdm import tqdm

if __name__ == "__main__":
    dataset = type("dataset", (), {})
    dataset.path = Path("data")
    dataset.items = list(dataset.path.glob("*"))
    dataset.labels = list(item for item in dataset.items if item.is_dir())
    
    output = type("output", (), {})
    output.path = dataset.path.parent / "data_rotated"
    
    print(f"Dataset")
    for key, value in dataset.__dict__.items():
        if key[:2] == "__": continue
        print(f"...\t{key:10s}: `{value}`")

    print(f"Preprocess...")
    for label in dataset.labels:
        if label.stem in ["Neg"]: continue # excluded labels
        print(f"Parsing `{label.stem}`")
        Path(f"{output.path}/{label.stem}").mkdir(parents=True, exist_ok=True)
        files = list(label.glob("*.txt"))
        images = list(label.glob("*.png"))
        i = 0
        for file, image in tqdm(zip(files, images), total = len(images), leave=False):
            file    = open(str(file),"r")
            content = file.read().splitlines()
            data    = [list(map(float,line.split('\t')[0:13])) for line in content]
            file.close()
            x_s = [[round(i) for i in x[0:7:2]] for x in data]
            y_s = [[round(i) for i in y[1:8:2]] for y in data]
            theta_s = [theta[8] for theta in data]
            lx_s = [lx[9] for lx in data]
            ly_s = [ly[10] for ly in data]
            width_s = [width[11] for width in data]
            height_s = [height[12] for height in data]
        
            img = cv2.imread(str(image), cv2.IMREAD_COLOR)
            
            for x, y, theta, width, height, lx, ly in zip(x_s, y_s, theta_s, width_s, height_s, lx_s, ly_s):
                
                top_left_x = min(x)
                top_left_y = min(y)
                bot_right_x = max(x)
                bot_right_y = max(y)
                firstPoint = (top_left_x, top_left_y)
                endPoint = (bot_right_x, bot_right_y)
                color = (255, 0, 0)
                result= img[top_left_y:bot_right_y, top_left_x:bot_right_x]
        
                width, height = result.shape[:2]
                # print(f"height {height} and width {width}")
                x = height / width
                if x >1 :
                    swapped_image = cv2.transpose(result)
                    if not cv2.imwrite(f'{output.path}/{label.stem}/image_{i:05d}.png',swapped_image):
                        raise Exception(f"Couldn't write `{output.path}/{label.stem}/image_{i:05d}.png`")
                    i +=1
                else:
                    if not cv2.imwrite(f'{output.path}/{label.stem}/image_{i:05d}.png',result):
                        raise Exception(f"Couldn't write `{output.path}/{label.stem}/image_{i:05d}.png`")
                    i +=1