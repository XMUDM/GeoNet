import torch
from schnetpack.datasets import MD17

# __all__ = ["build_mse_loss", "build_mse_loss_with_forces"]


class LossFnError(Exception):
    pass


def build_mse_loss(properties, loss_tradeoff=None):
    """
    Build the mean squared error loss function.

    Args:
        properties (list): mapping between the model properties and the
            dataset properties
        loss_tradeoff (list or None): multiply loss value of property with tradeoff
            factor

    Returns:
        mean squared error loss function

    """
    if loss_tradeoff is None:
        loss_tradeoff = [1] * len(properties)
    if len(properties) != len(loss_tradeoff):
        raise LossFnError("loss_tradeoff must have same length as properties!")

    def loss_fn(batch, result):
        loss = 0.0
        for prop, factor in zip(properties, loss_tradeoff):
            #diff = batch[prop] - torch.sum(batch['group_energy'], dim=1) - result[prop]
            if prop == 'group_energy':
                diff = batch[prop] - result[prop]
                diff = diff ** 2
                err_sq = factor * torch.mean(diff)
                loss += err_sq
            elif prop == 'total_rho':
                mesh_size = result[prop].shape[-1]
                norm = mesh_size**3
                diff = batch[prop][:,:norm] - result[prop].reshape(-1,norm)
                diff = diff **2
                err_sq = factor * torch.sum(diff)
                loss += err_sq
            elif prop == 'diff_U0_group':
                diff = batch[prop] - result[prop]
                diff = diff ** 2
                err_sq = factor * torch.mean(diff)
                loss += err_sq
        return loss, diff

    return loss_fn


def mse_loss(pred,target):
    diff = pred-target
    diff = diff ** 2
    loss = torch.mean(diff)
    return loss,diff

def build_mse_loss_with_forces(rho_tradeoff, with_forces, conformer_loss_weight=0.1, contrastive_loss_weight=0.01):
    """
    Build the mean squared error loss function.

    Args:
        rho_tradeoff (float): 控制能量和力之间的损失权衡
        with_forces (bool): 是否计算力的损失
        conformer_loss_weight (float): 构象一致性损失的权重
        contrastive_loss_weight (float): 对比学习损失的权重

    Returns:
        mean squared error loss function

    """

    def loss_with_forces(data, result):
        # compute the mean squared error on the energies
        diff_energy = data["energy"]-result["energy"]
        err_sq_energy = torch.mean(diff_energy ** 2)
        # compute the mean squared error on the forces
        diff_forces = data["forces"]-result["forces"]
        err_sq_forces = torch.mean(diff_forces ** 2)
        
        # 添加构象一致性损失（如果存在）
        conformer_loss = 0.0
        if "conformer_loss" in result:
            conformer_loss = result["conformer_loss"]
            # 记录这个损失以便于监控
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                # 同步所有进程中的构象损失
                torch.distributed.all_reduce(conformer_loss)
                if torch.distributed.get_world_size() > 0:
                    conformer_loss = conformer_loss / torch.distributed.get_world_size()
        
        # 添加对比学习损失（如果存在）
        contrastive_loss = 0.0
        if "contrastive_loss" in result and result["contrastive_loss"] is not None:
            contrastive_loss = result["contrastive_loss"]
            # 确保对比损失是tensor类型
            if not isinstance(contrastive_loss, torch.Tensor):
                contrastive_loss = torch.tensor(contrastive_loss, device=data["energy"].device)
            # 记录这个损失以便于监控
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                # 同步所有进程中的对比损失
                torch.distributed.all_reduce(contrastive_loss)
                if torch.distributed.get_world_size() > 0:
                    contrastive_loss = contrastive_loss / torch.distributed.get_world_size()
        
        # 构建组合损失函数
        err_sq = (rho_tradeoff * err_sq_energy + 
                 (1 - rho_tradeoff) * err_sq_forces + 
                 conformer_loss_weight * conformer_loss + contrastive_loss)

        return err_sq
    
    def loss_for_energy(data, result):
        # compute the mean squared error on the energies
        diff_energy = data["energy"]-result["energy"]
        err_sq_energy = torch.mean(diff_energy ** 2)

        # 添加构象一致性损失（如果存在）
        conformer_loss = 0.0
        if "conformer_loss" in result:
            conformer_loss = result["conformer_loss"]
            # 记录这个损失以便于监控
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                # 同步所有进程中的构象损失
                torch.distributed.all_reduce(conformer_loss)
                if torch.distributed.get_world_size() > 0:
                    conformer_loss = conformer_loss / torch.distributed.get_world_size()
        
        # 添加对比学习损失（如果存在）
        contrastive_loss = 0.0
        if "contrastive_loss" in result and result["contrastive_loss"] is not None:
            contrastive_loss = result["contrastive_loss"]
            # 确保对比损失是tensor类型
            if not isinstance(contrastive_loss, torch.Tensor):
                contrastive_loss = torch.tensor(contrastive_loss, device=data["energy"].device)
            # 记录这个损失以便于监控
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                # 同步所有进程中的对比损失
                torch.distributed.all_reduce(contrastive_loss)
                if torch.distributed.get_world_size() > 0:
                    contrastive_loss = contrastive_loss / torch.distributed.get_world_size()
        
        # 构建组合损失函数
        err_sq = (err_sq_energy + 
                 conformer_loss_weight * conformer_loss + contrastive_loss)

        return err_sq
        
    if with_forces:
        return loss_with_forces
    else:
        return loss_for_energy

