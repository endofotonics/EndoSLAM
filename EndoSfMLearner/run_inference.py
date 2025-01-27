import torch

# from imageio import imread, imsave
from skimage import transform, io, color
# from scipy.misc import imresize
import numpy as np
from pathlib import Path
import argparse
from tqdm.auto import tqdm

from models import DispResNet
from utils import tensor2array

parser = argparse.ArgumentParser(description='Inference script for DispNet learned with \
                                 Structure from Motion Learner inference on KITTI Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--output-disp", action='store_true', help="save disparity img")
parser.add_argument("--output-depth", action='store_true', help="save depth img")
parser.add_argument("--pretrained", required=True, type=str, help="pretrained DispResNet path")
parser.add_argument("--img-height", default=256, type=int, help="Image height")
parser.add_argument("--img-width", default=832, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")

parser.add_argument("--dataset-list", default=None, type=str, help="Dataset list file")
parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
parser.add_argument("--output-dir", default='output', type=str, help="Output directory")
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")
parser.add_argument('--resnet-layers', required=True, type=int, default=18, choices=[18, 50],
                    help='depth network architecture.')

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'~~~Using {device}~~~')


@torch.no_grad()
def main():
    args = parser.parse_args()
    if not(args.output_disp or args.output_depth):
        print('You must at least output one value !')
        return

    disp_net = DispResNet(args.resnet_layers, False).to(device)
    weights = torch.load(args.pretrained, map_location=torch.device(device))
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    if args.dataset_list is not None:
        with open(args.dataset_list, 'r') as f:
            test_files = [dataset_dir/file for file in f.read().splitlines()]
    else:
        test_files = sum([list(dataset_dir.glob(f'*.{ext}')) for ext in args.img_exts], [])

    print(f'{len(test_files)} files to test')

    for file in tqdm(test_files):

        img = io.imread(file).astype(np.float32)

        h, w, _ = img.shape
        if (not args.no_resize) and (h != args.img_height or w != args.img_width):
            img = transform.resize(img, (args.img_height, args.img_width)).astype(np.float32)
        img = np.transpose(img, (2, 0, 1))

        tensor_img = torch.from_numpy(img).unsqueeze(0)
        tensor_img = ((tensor_img/255 - 0.45)/0.225).to(device)

        output = disp_net(tensor_img)[0]

        # file_path, file_ext = file.relpath(args.dataset_dir).splitext()
        file_name = Path('-'.join(file.relative_to(args.dataset_dir).parts))

        if args.output_disp:
            disp = (255*tensor2array(output, max_value=None, colormap='bone')).astype(np.uint8)
            io.imsave(
                output_dir/f'{file_name.stem}_disp{file_name.suffix}',
                (255*color.rgba2rgb(np.transpose(disp, (1, 2, 0)))).astype(np.uint8)
            )
        if args.output_depth:
            depth = 1/output
            depth = (255*tensor2array(depth, max_value=10, colormap='rainbow')).astype(np.uint8)
            io.imsave(
                output_dir/f'{file_name.stem}_depth{file_name.suffix}',
                (255*color.rgba2rgb(np.transpose(depth, (1, 2, 0)))).astype(np.uint8)
            )


if __name__ == '__main__':
    main()
