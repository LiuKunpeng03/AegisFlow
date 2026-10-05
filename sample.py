import os

# os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"

from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import os
import torch
import numpy as np

from RectifiedFlow.models import ddpm, ncsnv2, ncsnpp
from RectifiedFlow.models import utils as mutils
from RectifiedFlow.models.ema import ExponentialMovingAverage
from RectifiedFlow.utils import restore_checkpoint
import RectifiedFlow.datasets as datasets
from utils.method_utils import save_img, generate_traj

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config",
    "RectifiedFlow/configs/celeba_hq_pytorch_rf_gaussian.py",
    "Rectified Flow Model configuration.",
    lock_config=True,
)
flags.DEFINE_integer("num_samples", 10, "number of samples to generate")
flags.DEFINE_string("output_dir", "sample", "output directory for samples")
flags.DEFINE_integer("seed", 42, "random seed")

model_path = "./pretrained.pth"


def main(argv):
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(FLAGS.seed)

    print(f"Generating {FLAGS.num_samples} samples...")

    os.makedirs(FLAGS.output_dir, exist_ok=True)

    inverse_scaler = datasets.get_data_inverse_scaler(FLAGS.config)

    score_model = mutils.create_model(FLAGS.config)
    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=FLAGS.config.model.ema_rate
    )
    state = dict(model=score_model, ema=ema, step=0)
    state = restore_checkpoint(model_path, state, device=FLAGS.config.device)
    ema.copy_to(score_model.parameters())
    model_fn = mutils.get_model_fn(score_model, train=False)

    N = 100
    batch_size = 1

    img_size = FLAGS.config.data.image_size
    num_channels = FLAGS.config.data.num_channels
    shape = (batch_size, num_channels, img_size, img_size)

    for sample_id in range(FLAGS.num_samples):
        print(f"Generating sample {sample_id + 1}/{FLAGS.num_samples}...")

        z0 = torch.randn(shape, device=FLAGS.config.device)

        traj = generate_traj(dynamics=model_fn, z0=z0, N=N)

        final_image = traj[-1]

        sample_path = os.path.join(FLAGS.output_dir, f"sample_{sample_id:03d}.png")
        save_img(inverse_scaler(final_image), path=sample_path)

        print(f"Saved sample to: {sample_path}")

    print(f"\nGenerated {FLAGS.num_samples} samples in '{FLAGS.output_dir}' folder")


if __name__ == "__main__":
    app.run(main)
