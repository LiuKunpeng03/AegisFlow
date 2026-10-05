import torch

def load_torch_file(path, map_location=None, trusted=False):
  load_kwargs = {}
  if map_location is not None:
    load_kwargs['map_location'] = map_location

  if trusted:
    try:
      return torch.load(path, weights_only=False, **load_kwargs)
    except TypeError:
      return torch.load(path, **load_kwargs)

  return torch.load(path, **load_kwargs)

def restore_checkpoint(ckpt_dir, state, device):
    loaded_state = torch.load(
        ckpt_dir,
        map_location=device,
        weights_only=False,
    )
    state["model"].load_state_dict(loaded_state["model"], strict=False)
    state["ema"].load_state_dict(loaded_state["ema"])
    state["step"] = loaded_state["step"]
    return state


def save_checkpoint(ckpt_dir, state):
    saved_state = {
        "optimizer": state["optimizer"].state_dict(),
        "model": state["model"].state_dict(),
        "ema": state["ema"].state_dict(),
        "step": state["step"],
    }
    torch.save(saved_state, ckpt_dir)
