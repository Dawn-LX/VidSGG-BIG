import logging
from itertools import chain

import torch
# from torch_geometric.data import Batch


class VidSGG_DataParallel(torch.nn.DataParallel):
    r"""Implements data parallelism at the module level.
    NOTE this class has been modified, original function code from 
    https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/nn/data_parallel.py
    
    This container parallelizes the application of the given :attr:`module` by
    splitting a list of :class:`torch_geometric.data.Data` objects and copying
    them as :class:`torch_geometric.data.Batch` objects to each device.
    In the forward pass, the module is replicated on each device, and each
    replica handles a portion of the input.
    During the backwards pass, gradients from each replica are summed into the
    original module.

    The batch size should be larger than the number of GPUs used.

    The parallelized :attr:`module` must have its parameters and buffers on
    :obj:`device_ids[0]`.

    .. note::

        You need to use the :class:`torch_geometric.data.DataListLoader` for
        this module.

    Args:
        module (Module): Module to be parallelized.
        device_ids (list of int or torch.device): CUDA devices.
            (default: all devices)
        output_device (int or torch.device): Device location of output.
            (default: :obj:`device_ids[0]`)
    """
    def __init__(self, module, device_ids=None, output_device=None):
        super(VidSGG_DataParallel, self).__init__(module, device_ids, output_device)
        self.src_device = torch.device("cuda:{}".format(self.device_ids[0]))

    def forward(self, data_list, gt_list):
        """"""
        assert len(data_list) == len(gt_list)
        if len(data_list) == 0:
            logging.warning('VidSGG_DataParallel received an empty data list, which '
                            'may result in unexpected behaviour.')
            return None

        if not self.device_ids or len(self.device_ids) == 1:  # Fallback
            data_list = [d.to(self.src_device) for d in data_list]
            gt_list = [g.to(self.src_device) for g in gt_list]
            return self.module(data_list,gt_list)

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device:
                raise RuntimeError(
                    ('Module must have its parameters and buffers on device '
                     '{} but found one of them on device {}.').format(
                         self.src_device, t.device))
        
        inputs = self.scatter(data_list, gt_list, self.device_ids)
        
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, None)
        return self.gather(outputs, self.output_device)

    def scatter(self,data_list,gt_list,device_ids):
        assert len(data_list) == len(gt_list)
        batch_size = len(data_list)
        num_devices = len(device_ids)
        if batch_size==4 and num_devices==2:
            return self.scatter_42(data_list,gt_list,device_ids)
        
        assert batch_size >= num_devices, "batch_size={} >= num_gpus={}".format(len(data_list),len(device_ids))
        assert batch_size % num_devices == 0  # TODO add the code to handle the situation `batch_size % num_devices != 0`

        # batch_list = [(data_list_1,gt_list_1),...,(data_list_N,gt_list_N)]
        batch_list = []
        n_per_device = batch_size // num_devices
        for i in range(num_devices):
            device_i = torch.device('cuda:{}'.format(device_ids[i]))
            s_idx = i*n_per_device
            e_idx = (i+1)*n_per_device
            data_list_i = [d.to(device_i) for d in  data_list[s_idx:e_idx]]
            gt_list_i =  [g.to(device_i) for g in  gt_list[s_idx:e_idx]]
            batch_list.append(
                (data_list_i,gt_list_i)
            )
        return batch_list 
    
    def scatter_42(self,data_list,gt_list,device_ids):
        data_list_1 = [data_list[0],data_list[3]]
        data_list_2 = [data_list[1],data_list[2]]
        gt_list_1 = [gt_list[0],gt_list[3]]
        gt_list_2 = [gt_list[1],gt_list[2]]

        device_1 = torch.device('cuda:{}'.format(device_ids[0]))
        device_2 = torch.device('cuda:{}'.format(device_ids[1]))
        

        data_list_1 = [d.to(device_1) for d in  data_list_1]
        gt_list_1 =  [g.to(device_1) for g in  gt_list_1]

        data_list_2 = [d.to(device_2) for d in  data_list_2]
        gt_list_2 =  [g.to(device_2) for g in  gt_list_2]

        batch_list = [(data_list_1,gt_list_1),(data_list_2,gt_list_2)]

        return batch_list

