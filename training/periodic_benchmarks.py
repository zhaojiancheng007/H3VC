import os


def evaluation_environment(local_rank: int) -> dict[str, str]:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    physical_device = (
        str(local_rank)
        if visible_devices is None
        else visible_devices.split(",")[local_rank]
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = physical_device
    environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(name, None)
    return environment

