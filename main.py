import os

# os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"

import json
from tqdm import tqdm
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import glob
import lpips


from RectifiedFlow.models import ddpm, ncsnv2, ncsnpp
from RectifiedFlow.models import utils as mutils
from RectifiedFlow.models.ema import ExponentialMovingAverage
import RectifiedFlow.datasets as datasets
from RectifiedFlow.utils import restore_checkpoint
from utils.methods import image_edit, sanitize_prompt_for_filename
from utils.method_utils import clip_semantic_loss

import torch


FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config",
    "RectifiedFlow/configs/celeba_hq_pytorch_rf_gaussian.py",
    "Rectified Flow Model configuration.",
    lock_config=True,
)

flags.DEFINE_string(
    "folder",
    "demo",
    "Folder containing images to be edited.",
)

flags.DEFINE_string(
    "method",
    "oc_flow",
    "Method to use for image editing.",
)

flags.DEFINE_string(
    "output_folder",
    "output",
    "Output folder for processed images.",
)

text_prompts = [
    "A photo of an old face.",
    "A photo of a sad face.",
    "A photo of a smiling face.",
    "A photo of an angry face.",
    "A photo of a face with curly hair.",
]

model_path = "/root/autodl-tmp/AegisFlow/wt/rectifiedflow.pth"


def main(argv):
    image_folder = f"{FLAGS.folder}"
    image_files = glob.glob(os.path.join(image_folder, "*.jpg"))

    # sort images by file names
    image_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    print(f"Found {len(image_files)} images in {image_folder}")

    output_folder = FLAGS.output_folder
    os.makedirs(output_folder, exist_ok=True)

    score_model = mutils.create_model(FLAGS.config)
    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=FLAGS.config.model.ema_rate
    )
    state = dict(model=score_model, ema=ema, step=0)
    state = restore_checkpoint(model_path, state, device=FLAGS.config.device)
    ema.copy_to(score_model.parameters())
    model_fn = mutils.get_model_fn(score_model, train=False)

    clip_loss_1 = clip_semantic_loss(
        FLAGS.config.device,
        alpha=1.0,
        inverse_scaler=datasets.get_data_inverse_scaler(FLAGS.config),
    )
    lpips_model = lpips.LPIPS(net="alex").to(FLAGS.config.device).eval()

    models = {
        "model_fn": model_fn,
        "clip_loss_1": clip_loss_1,
        "lpips_model": lpips_model,
    }

    for image_path in tqdm(image_files, desc="Processing images..."):
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        image_method_folder = os.path.join(output_folder, image_name, FLAGS.method)
        os.makedirs(image_method_folder, exist_ok=True)

        metrics = {}

        for i, prompt in enumerate(text_prompts):

            print(f"\nProcessing {image_name} with prompt: '{prompt}'")

            metric = image_edit(
                FLAGS.config, models, image_path, prompt, image_method_folder
            )

            metric['peak_memory'] = torch.cuda.max_memory_allocated() / 1024 / 1024

            metrics[sanitize_prompt_for_filename(prompt)] = metric

        # dump to json
        with open(os.path.join(image_method_folder, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

    print("\nProcessing completed!")


if __name__ == "__main__":
    app.run(main)
