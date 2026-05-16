import argparse
import os
from PIL import Image
from ultralytics import YOLO


def save_overlay(results, output_dir="output_images"):
    os.makedirs(output_dir, exist_ok=True)
    for i, result in enumerate(results):
        img_bgr = result.plot()  # numpy BGR array with boxes/keypoints drawn
        img_rgb = img_bgr[..., ::-1]
        path = os.path.join(output_dir, f"result_{i}.jpg")
        Image.fromarray(img_rgb).save(path)
        print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pose", action="store_true", help="Use pose model")
    args = parser.parse_args()

    if args.pose:
        model = YOLO("yolo26n-pose.pt")
        results = model("https://ultralytics.com/images/bus.jpg")
        for result in results:
            xy = result.keypoints.xy
            xyn = result.keypoints.xyn
            kpts = result.keypoints.data
            print("xy:", xy)
            print("xyn:", xyn)
            print("kpts:", kpts)
    else:
        model = YOLO("yolo26n.pt")
        model.export(format="coreml")
        coreml_model = YOLO("yolo26n.mlpackage")
        results = coreml_model("https://ultralytics.com/images/bus.jpg")

    save_overlay(results)


if __name__ == "__main__":
    main()
