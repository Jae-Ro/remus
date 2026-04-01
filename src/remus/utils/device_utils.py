from dataclasses import dataclass
from typing import Optional

from remus.utils.dataclass_utils import BaseSerial


@dataclass
class GPUDevInfo(BaseSerial):
    devices: list[str | int]
    main_device: int | str
    use_gpu: bool
    is_multi_gpu: bool


def get_gpu_device_info(
    devices: Optional[list[int | str] | str] = None,
) -> GPUDevInfo:
    """Utility function to get gpu device information.

    **IMPORTANT:** wherever you call this function, that process becomes the CUDA context

    :param devices: either list of device ids as int/str OR a single string, defaults to None
    :return: instance of GPUDevInfo
    """
    import torch

    # normalize device ids
    if devices is None:
        devices = (
            [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available()
            else ["cpu"]
        )

    else:
        if isinstance(devices, str):
            devices = [devices]

        devices = [f"cuda:{d}" if isinstance(d, int) else d for d in devices]

    # determine if gpu vs cpu
    # if gpu, single vs multi
    use_gpu = any("cuda" in d for d in devices)
    main_device = devices[0] if use_gpu else "cpu"
    is_multi_gpu = len(devices) > 1

    return GPUDevInfo(
        devices=devices,
        main_device=main_device,
        use_gpu=use_gpu,
        is_multi_gpu=is_multi_gpu,
    )
