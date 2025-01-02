import os
from pipeline import FabricDiffusionPipeline
import argparse


def run_flatten_texture(pipeline, warp_dataset_path, output_path=None, n_samples=3):
    os.makedirs(os.path.join(output_path), exist_ok=True)
    all_image_names = os.listdir(warp_dataset_path)
    for image_name in all_image_names:
        texture_name = image_name.split('.')[0]
        texture_patch = pipeline.load_patch_data(os.path.join(warp_dataset_path, image_name))
        gen_imgs = pipeline.flatten_texture(texture_patch, n_samples=n_samples)
        for i, gen_img in enumerate(gen_imgs):
            gen_img.save(os.path.join(output_path, f'{texture_name}_gen_{i}.png'))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to run the model"
    )
    parser.add_argument(
        "--texture_checkpoint", default=None, type=str, help="Path to the texture model checkpoint"
    )
    parser.add_argument(
        "--print_checkpoint", default=None, type=str, help="Path to the logo model checkpoint"
    )
    parser.add_argument(
        "--src_dir", default='./data/texture_examples', type=str, help="Path to the input image directory"
    )
    parser.add_argument(
        "--save_dir", type=str, default='./outputs/texture',  help="Directory to save the output"
    )
    parser.add_argument(
        "--n_samples", type=int, default=3, help="Number of generated images per input"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    device = args.device
    texture_checkpoint = args.texture_checkpoint
    print_checkpoint = args.print_checkpoint
    src_dir = args.src_dir
    save_dir = args.save_dir
    
    pipeline = FabricDiffusionPipeline(device, texture_checkpoint, print_checkpoint=print_checkpoint)

    os.makedirs(save_dir, exist_ok=True)
    run_flatten_texture(pipeline, src_dir, output_path=save_dir, n_samples=args.n_samples)
