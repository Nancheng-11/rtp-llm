import logging
from enum import Enum
from typing import Dict, Optional

import torch
from torch.distributed.distributed_c10d import AllreduceOptions, ReduceOp

from rtp_llm.models_py.distributed.symm_mem import (
    TorchSymmMemCommunicator,
    _init_symm_mem_communicator,
)


class Group(Enum):
    DP = "DP"
    TP = "TP"
    DP_AND_TP = "DP_AND_TP"


try:
    from rtp_llm.ops.compute_ops import RtpProcessGroup, RtpProcessGroupType

    _group_map: Dict[Group, RtpProcessGroup] = {}

    _symm_mem_comm: Optional[TorchSymmMemCommunicator] = None

    def _get_symm_mem_communicator(
        tp_group: torch.distributed.ProcessGroup, buffer: torch.Tensor = None
    ) -> Optional[TorchSymmMemCommunicator]:
        """Get or initialize TorchSymmMemCommunicator (lazy initialization)."""
        global _symm_mem_comm
        if _symm_mem_comm is None:
            logging.info(f"TorchSymmMemCommunicator: get_symm_mem_communicator")
            _symm_mem_comm = _init_symm_mem_communicator(tp_group, buffer)
        return _symm_mem_comm

    def _group_type_convert(group: Group) -> RtpProcessGroupType:
        if group == Group.DP:
            return RtpProcessGroupType.DP_GROUP
        elif group == Group.TP:
            return RtpProcessGroupType.TP_GROUP
        elif group == Group.DP_AND_TP:
            return RtpProcessGroupType.DP_AND_TP_GROUP
        raise ValueError(f"Invalid group: {group}")

    def _get_group(group: Group) -> RtpProcessGroup:
        if group not in _group_map:
            _group_map[group] = RtpProcessGroup(_group_type_convert(group))
        return _group_map[group]

    def send(tensor: torch.Tensor, dst: int, group: Group = Group.DP_AND_TP) -> None:
        _get_group(group).send([tensor], dst)

    def recv(
        tensor: torch.Tensor, src: int, group: Group = Group.DP_AND_TP
    ) -> torch.Tensor:
        return _get_group(group).recv([tensor], src)

    def broadcast(
        tensor: torch.Tensor, src: int, group: Group = Group.DP_AND_TP
    ) -> None:
        _get_group(group).broadcast([tensor], src)

    def all_reduce(
        tensor: torch.Tensor, group: Group = Group.DP_AND_TP
    ) -> torch.Tensor:
        if group == Group.TP:
            from rtp_llm.distribute.gang_server import (
                g_symm_mem_buffer,
                g_symm_mem_group,
            )

            symm_mem_comm = _get_symm_mem_communicator(
                g_symm_mem_group, g_symm_mem_buffer
            )
            if (
                symm_mem_comm is not None
                and symm_mem_comm.should_torch_symm_mem_allreduce(tensor)
            ):
                result = symm_mem_comm.all_reduce(tensor)
                if result is not None:
                    return result
        return _get_group(group).all_reduce([tensor])[0]

    def all_gather(
        tensor: torch.Tensor, group: Group = Group.DP_AND_TP
    ) -> torch.Tensor:
        return _get_group(group).all_gather([tensor])[0]

except ImportError:
    logging.info("RtpProcessGroup not available, skipped. Defining dummy functions.")

    def _raise_error_on_call(*args, **kwargs):
        raise ImportError(
            "RtpProcessGroup functions are not available because 'libth_transformer' could not be imported."
        )

    send = recv = broadcast = all_reduce = all_gather = _raise_error_on_call
