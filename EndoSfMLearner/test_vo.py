import torch

# from imageio import imread , imsave
from skimage.io import imread
from skimage.transform import resize as imresize
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

from inverse_warp import pose_vec2mat
# from scipy.ndimage import zoom

from inverse_warp import *

import models
# from utils import tensor2array

parser = argparse.ArgumentParser(description='Script for visualizing depth map and masks',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--pretrained-posenet", required=True, type=str, help="pretrained PoseNet path")
parser.add_argument("--img-height", default=256, type=int, help="Image height")
parser.add_argument("--img-width", default=832, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")

parser.add_argument("--dataset-dir", type=str, help="Dataset directory")
parser.add_argument("--output-dir", type=str, help="Output directory for saving predictions in a big 3D numpy file")
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")
parser.add_argument("--rotation-mode", default='euler', choices=['euler', 'quat'], type=str)

# parser.add_argument("--sequence", default='09', type=str, help="sequence to test")
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'~~~Using {device}~~~')

def load_tensor_image(filename, args):
    img = imread(filename).astype(np.float32)
    h, w, _ = img.shape
    if (not args.no_resize) and (h != args.img_height or w != args.img_width):
        img = imresize(img, (args.img_height, args.img_width)).astype(np.float32)
    img = np.transpose(img, (2, 0, 1))
    tensor_img = ((torch.from_numpy(img).unsqueeze(0)/255-0.45)/0.225).to(device)
    return tensor_img


@torch.no_grad()
def main():
    args = parser.parse_args()

    weights_pose = torch.load(args.pretrained_posenet, map_location=torch.device(device))
    pose_net = models.PoseResNet().to(device)
    pose_net.load_state_dict(weights_pose['state_dict'], strict=False)
    pose_net.eval()

    image_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    test_files = sorted(
        sum([list(image_dir.glob(f'*.{ext}')) for ext in args.img_exts], [])
    )

    print(f'{len(test_files)} files to test')

    global_pose = np.eye(4)
    poses = [global_pose[0:3, :].reshape(1, 12)]

    n = len(test_files)
    tensor_img1 = load_tensor_image(test_files[0], args)

    for iter in tqdm(range(n - 1)):

        tensor_img2 = load_tensor_image(test_files[iter+1], args)

        pose = pose_net(tensor_img1, tensor_img2)

        pose_mat = pose_vec2mat(pose).squeeze(0).cpu().numpy()
        pose_mat = np.vstack([pose_mat, np.array([0, 0, 0, 1])])
        global_pose = global_pose @ np.linalg.inv(pose_mat)

        poses.append(global_pose[0:3, :].reshape(1, 12))

        # update
        tensor_img1 = tensor_img2

    poses = np.concatenate(poses, axis=0)
    # filename = output_dir/(args.sequence + ".txt")
    filename = output_dir/(image_dir.name + '.txt')
    np.savetxt(filename, poses, delimiter=' ', fmt='%1.8e')
    print('Exported as', filename.absolute())


if __name__ == '__main__':
    main()
