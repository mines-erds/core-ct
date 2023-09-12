from importers import dicom
from glob import glob
from PIL import Image as im
import numpy as np
import argparse

def main() -> None:
    parser = argparse.ArgumentParser(description="Test loading of a dicom dataset")
    parser.add_argument("dir", metavar="DIRECTORY", type=str, help="Path to the directory containing dicom dataset")

    args = parser.parse_args()
    dir: str = args.dir
    
    dicom_from_dir(dir)
    glob_str = dir + "/*"
    dicom_from_files(glob(glob_str))

def dicom_from_dir(dir: str) -> None:
    core = dicom(dir=dir)
    test_core(core, "dicom_from_dir.png")

def dicom_from_files(files: list[str]) -> None:
    core = dicom(files=files)
    test_core(core, "dicom_from_files.png")

def test_core(core, output: str) -> None:
    print(core.shape)
    slice = core[:, :, 0]
    print(slice.shape)
    max: float = np.max(slice)
    min: float = np.min(slice)
    print(max)
    print(min)
    offset = abs(min)
    norm_max = max + offset
    for i in range(len(slice)):
        for j in range(len(slice[0])):
            slice[i][j] = int(((slice[i][j] + offset) / norm_max) * 255)
    picture = im.fromarray(slice)
    picture = picture.convert("L")
    picture.save(output)

if __name__ == "__main__":
    main()