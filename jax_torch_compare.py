import torch


def jax_path_to_torch_key(path) -> str:
    path = [x for x in path if x != "layers"]
    if path[-1] in ("kernel", "scale", "embedding"):
        path[-1] = "weight"
    return ".".join("%s" % x for x in path)


def load_torch_model_into_jax(torch_model_path: str, jax_state):
    torch_state = torch.load(torch_model_path)

    jax_flat_state = {
        jax_path_to_torch_key(path): value for path, value in jax_state.flat_state()
    }

    assert len(jax_flat_state) == len(torch_state)

    for k in jax_flat_state:
        jax_shape = jax_flat_state[k].value.shape
        value = torch_state[k].numpy()
        if len(jax_shape) == 2 and k != "encoder.weight":
            value = value.transpose()
        jax_flat_state[k].value = value

    return jax_state
