import cv2
import numpy as np

from multiprocessing import Pool
import logging

from tqdm import tqdm
from pathlib import Path

import os
from dotenv import load_dotenv
load_dotenv()

data_folder = Path( os.environ["DATA_FOLDER"] )
imgs_folder = data_folder / "images"

logger = logging.getLogger(__name__)
logging.basicConfig(filename=".logs/mp_imgs_setup.log", level=logging.INFO,
                    format="%(levelname)s:%(name)s:%(asctime)s:%(message)s",
                    datefmt="%I:%M:%S %p")



def save_imgs_parition(index: int) -> None:
    index = str(index).zfill(2)
    logger.info(f"working on folder `{index}`")

    fodler_path = imgs_folder / index
    folder_content = sorted(fodler_path.iterdir())

    imgs_L = []
    imgs_ab = []

    for file in folder_content:
        img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2LAB)
        # BGR -> Lab

        img = cv2.resize(img, (224, 224))
        # shape -> (224, 224)

        # img = img.astype(np.float32) / 255.0
        # # normalize all pixels to be in [0, 1]  (if not `astype`, dtype=float64)

        # img.shape = (224, 224, 3)
        imgs_L.append(img[:, :, 0])
        imgs_ab.append(img[:, :, 1:])
    
    stack_L = np.stack(imgs_L)
    stack_ab = np.stack(imgs_ab)

    np.save(f"data/L/{index}.npy", stack_L)
    np.save(f"data/ab/{index}.npy", stack_ab)


    logger.info(f"work on folder `{index}` done")




if __name__ == "__main__":
    print("converion start")
    logger.warning("converion start")

    with Pool(12) as p:
        pb = tqdm(total=100, colour="#6ac856")

        processing = p.imap_unordered(save_imgs_parition, range(100),
                                      chunksize=1)  # 4
        
        for proces in processing:
            pb.update(1)
            pb.refresh()

        pb.close()


    print("all converted")
    logger.warning("all converted")
