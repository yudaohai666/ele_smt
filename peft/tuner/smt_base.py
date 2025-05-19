import importlib
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union
from deepspeed.compression.helper import recursive_getattr, recursive_setattr
from collections import defaultdict
import heapq

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D


Block_dimension = 256





class SMTModel(torch.nn.Module):
    """
    Creates Low Rank Adapter (Lora) model from a pretrained transformers model.

    Args:
        model ([`transformers.PreTrainedModel`]): The model to be adapted
        config ([`SMTConfig`]): The configuration of the SMT model.

    Returns:
        `torch.nn.Module`: The SMT model.

    Example::

        from transformers import AutoModelForSeq2SeqLM, LoraConfig
        from peft import LoraModel, LoraConfig
        config = LoraConfig(
            peft_type="LORA", task_type="SEQ_2_SEQ_LM", r=8, lora_alpha=32, target_modules=["q", "v"],
            lora_dropout=0.01, )
        model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        attention_grads = {}
        mlp_grads = {}
        lora_model = LoraModel(config, model, attention_grads, mlp_grads)

    **Attributes**:
        - **model** ([`transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`SMTConfig`]): The configuration of the SMT model.
        - **attention_grads** (['dict']): dictionary of attention gradient information.
        - **mlp_grads** (['dict']):  dictionary of mlp gradient information.
    """

    def __init__(self, config, model, attention_grads, mlp_grads):
        super().__init__()
        self.peft_config = config
        self.model = model
        selected_submatrix_mlp = {}
        selected_submatrix_attention = {}
        if self.peft_config.num_submatrix_mlp > 0:
            selected_submatrix_mlp = select_submatrix_based_on_grads(mlp_grads,
                                                                 self.peft_config.num_submatrix_mlp,
                                                                 selection_strategy=self.peft_config.selection_strategy,
                                                                 calculate_strategy=self.peft_config.calculate_strategy,
                                                                 model=self.peft_config.model_name_or_path)
        if self.peft_config.num_submatrix_attn > 0:
            selected_submatrix_attention = select_submatrix_based_on_grads(attention_grads,
                                                                           self.peft_config.num_submatrix_attn,
                                                                           selection_strategy=self.peft_config.selection_strategy,
                                                                           calculate_strategy=self.peft_config.calculation_strategy,
                                                                           model=self.peft_config.model_name_or_path)
        self.model = mark_only_smt_as_trainable(self.model, selected_submatrix_mlp, selected_submatrix_attention)  
        # self.model = mark_only_smt_as_trainable(self.model.module, selected_submatrix_mlp, selected_submatrix_attention)
        self.model = self.convert_linear_layer_to_matrix_sparsity(self.model, selected_submatrix_mlp, selected_submatrix_attention)
        self.print_trainable_parameters()
        # 8-bit quantization function not implemented yet
        # self._find_and_replace()
        self.forward = self.model.forward

    def convert_linear_layer_to_matrix_sparsity(self, model, selected_submatrix, selected_submatrix_attention,
                                                part_module_name=['.layers']):
        """
        将模型中特定的线性层转换为矩阵稀疏层。

        Args:
            model (torch.nn.Module): 待处理的 PyTorch 模型。
            selected_submatrix (dict): 选中的 MLP 子矩阵信息，键为 (module_name, layer_number) 元组，值为子矩阵索引列表。
            selected_submatrix_attention (dict): 选中的自注意力子矩阵信息，键为 (module_name, layer_number) 元组，值为子矩阵索引列表。
            part_module_name (list, optional): 模块名称中需要包含的部分，用于筛选模块，默认为 ['.layers']。

        Returns:
            torch.nn.Module: 处理后的模型。
        """
        # 编译正则表达式，用于从模块名称中提取层编号
        pattern = re.compile(r'model\.layers\.(\d+)\.')

        replace_name = []
        # 遍历模型中的所有模块，获取模块名称和对应的模块实例
        for name, module in model.named_modules():
            # 筛选出符合条件的线性层：是 nn.Linear 类型且模块名称包含 part_module_name 中的部分
            if isinstance(module, nn.Linear) and any(part in name for part in part_module_name):
                # print(f"convert {name} to LoRA")
                replace_name.append(name)
        # 遍历需要替换的模块名称列表
        for name in replace_name:
            if "mlp" in name:
                # 通过递归方式获取指定名称的模块
                module = recursive_getattr(model, name)
                # 若模块的权重需要计算梯度
                if module.weight.requires_grad:
                    module_name = 'gate_proj' if 'gate_proj' in name else 'up_proj' if 'up_proj' in name else 'down_proj'
                    # 使用正则表达式匹配层编号
                    match = pattern.search(name)
                    # 若匹配成功，提取层编号，否则为 None
                    layer_number = int(match.group(1)) if match else None

                     # 从 selected_submatrix 中获取需要计算梯度的子矩阵索引列表
                    index_list = selected_submatrix[(module_name, layer_number)]

                    # 创建一个矩阵稀疏层实例
                    tmp = LinearLayer_MatrixSparsity(
                        module.weight,
                        bias=None,
                        index_list=index_list).to(module.weight.device).to(module.weight.dtype)
                    # 通过递归方式将原模块替换为新创建的矩阵稀疏层
                    recursive_setattr(model, name, tmp)
            if "self_attn" in name:
                module = recursive_getattr(model, name)
                if module.weight.requires_grad:
                    module_name = 'q_proj' if 'q_proj' in name else 'k_proj' if 'k_proj' in name else 'v_proj' if 'v_proj' in name else None
                    match = pattern.search(name)
                    layer_number = int(match.group(1)) if match else None

                    # index_list: list of index which require_grad, need to pass into Linear
                    index_list = selected_submatrix_attention[(module_name, layer_number)]

                    tmp = LinearLayer_MatrixSparsity(
                        module.weight,
                        bias=None,
                        index_list=index_list).to(module.weight.device).to(module.weight.dtype)
                    recursive_setattr(model, name, tmp)

        return model

    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        # for param in self.model.module.parameters():
        for param in self.model.parameters():
        # for param in self.model.parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    def _get_submodules(self, key):
        parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = self.model.get_submodule(key)
        return parent, target, target_name

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight

        #
        with torch.no_grad():
            magnitude = (torch.linalg.norm(new_module.weight.detach(), dim=1)).unsqueeze(1).detach()
            new_module.weight_m_wdecomp.weight.copy_(magnitude)

        if old_module.bias is not None:
            new_module.bias = old_module.bias
        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name or "weight_m_wdecomp" in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    @property
    def modules_to_save(self):
        return None

    def get_peft_config_as_dict(self, inference: bool = False):
        config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(self.peft_config).items()}
        if inference:
            config["inference_mode"] = True
        return config


def mark_only_smt_as_trainable(model, select_parameters, select_attention_parameters):
    """
    将模型中特定模块的参数设置为可训练，其余参数设置为不可训练。

    Args:
        model (torch.nn.Module): 待处理的 PyTorch 模型。
        select_parameters (dict): 字典，键为 (module_name, layer_number) 元组，
            用于指定 MLP 层中需要设置为可训练的参数。
        select_attention_parameters (dict): 字典，键为 (module_name, layer_number) 元组，
            用于指定自注意力层中需要设置为可训练的参数。

    Returns:
        torch.nn.Module: 处理后的模型。
    """
    # selected_parameters: (module_name, layer_number, head_number)
    # model = convert_selected_sau_to_linear_layer(model, select_parameters, exclude)
    pattern = re.compile(r'model\.layers\.(\d+)\.')
    for name, param in model.named_parameters():
        if "mlp" in name:
            module_name = 'gate_proj' if 'gate_proj' in name else 'up_proj' if 'up_proj' in name else 'down_proj'
            match = pattern.search(name)
            layer_number = int(match.group(1)) if match else None
            if (module_name, layer_number) in select_parameters.keys():
                param.requires_grad = True
                # print(f"Layer set to grad = True:{name}")

                # print("selected grad True layer")
                # print(module_name, layer_number)
            else:
                param.requires_grad = False
                # print(f"Layer set to grad = Flase:{name}")

        elif "self_attn" in name:
            module_name = 'q_proj' if 'q_proj' in name else 'k_proj' if 'k_proj' in name else 'v_proj' if 'v_proj' in name else None
            match = pattern.search(name)
            layer_number = int(match.group(1)) if match else None
            if (module_name, layer_number) in select_attention_parameters.keys():
                param.requires_grad = True
                print(f"Layer set to grad = True:{name}")


            else:
                param.requires_grad = False
                print(f"Layer set to grad = Flase:{name}")

        else:
            param.requires_grad = False
            print(f"Layer set to grad = False:{name}")

    return model



class LinearLayer_MatrixSparsity(torch.nn.Module):
    # an simple implementation of matrix sparsity
    # for now only support Linear Layer
    def __init__(self,
                 weight,
                 bias=None,
                 index_list = []):
        """
        初始化 LinearLayer_MatrixSparsity 类的实例。

        Args:
            weight (torch.Tensor): 线性层的权重张量。
            bias (torch.Tensor, optional): 线性层的偏置张量，默认为 None。
            index_list (list, optional): 选中的子矩阵索引列表，每个元素为 (row, col) 元组，默认为空列表。
        """
        super(LinearLayer_MatrixSparsity, self).__init__()
        self.weight = weight
        self.weight.requires_grad = False
        self.bias = bias
        self.index_list = index_list

        # 初始化一个空的张量，用于存储选中的子矩阵的权重
        self.selected_weight = torch.empty(len(index_list) * Block_dimension, Block_dimension,dtype=self.weight.data.dtype,
                                  device=self.weight.data.device)

        # 遍历选中的子矩阵索引列表
        for i in range(len(index_list)):
            index = index_list[i]
            self.selected_weight[i * Block_dimension: i * Block_dimension + Block_dimension, :] = self.weight.data[index[0] * Block_dimension: index[0] * Block_dimension + Block_dimension, index[1] * Block_dimension: index[1] * Block_dimension + Block_dimension]

        # 将选中的子矩阵权重张量设置为可训练
        self.selected_weight.requires_grad = True
        # 将选中的子矩阵权重张量转换为 nn.Parameter，以便参与模型训练
        self.selected_weight = nn.Parameter(self.selected_weight)


        self.fn = linearZ.apply

    def forward(self, x):
        """
        前向传播：将可训练的子矩阵权重更新到原始权重，并使用自定义函数计算输出。

        Args:
            x (torch.Tensor): 输入张量，形状通常为 [batch_size, seq_len, hidden_size]

        Returns:
            torch.Tensor: 前向传播的输出结果
        """
        # 遍历所有选中的子矩阵索引，将可训练的 selected_weight 更新到原始权重的对应位置
        for i in range(len(self.index_list)):
            index = self.index_list[i]
            # 将 selected_weight 中第 i 个子矩阵的权重复制到原始权重的对应块位置
            # 原始权重的行块范围：index[0]*Block_dimension 到 (index[0]+1)*Block_dimension
            # 原始权重的列块范围：index[1]*Block_dimension 到 (index[1]+1)*Block_dimension
            # self.selected_weight[i * Block_dimension: i * Block_dimension + Block_dimension, :] = self.weight.data[index[0] * Block_dimension: index[0] * Block_dimension + Block_dimension, index[1] * Block_dimension: index[1] * Block_dimension + Block_dimension]
            self.weight.data[index[0] * Block_dimension: index[0] * Block_dimension + Block_dimension, index[1] * Block_dimension: index[1] * Block_dimension + Block_dimension] = self.selected_weight[i * Block_dimension: i * Block_dimension + Block_dimension, :]

        # 调用自定义自动微分函数 linearZ 执行前向计算（使用更新后的原始权重）
        x = self.fn(x,  self.selected_weight, self.index_list, self.weight)
        return x

class linearZ(torch.autograd.Function):
    # only support batch size D=3 now, for batch size = 1, need to add mm. operation.
    @staticmethod
    def forward(ctx, input, selected_weight, matrix_index_list, weight):
        """
        前向传播计算输出。

        Args:
            ctx (torch.autograd.function.BackwardCFunction): 上下文对象，用于存储反向传播所需信息。
            input (torch.Tensor): 输入张量，形状通常为 [batch_size, seq_len, hidden_size]。
            selected_weight (torch.Tensor): 选中的子矩阵权重，形状为 [num_blocks*Block_dimension, Block_dimension]。
            matrix_index_list (list): 选中的子矩阵索引列表，每个元素为 (row, col) 元组。
            weight (torch.Tensor): 原始完整权重张量，形状为 [hidden_size, hidden_size]（或类似结构）。

        Returns:
            torch.Tensor: 前向传播输出结果。
        """
        input_list = []
        for index in matrix_index_list:
            input_list.append(input[:, :, index[1]*Block_dimension: index[1]*Block_dimension+Block_dimension])
        # save for backward may only support tensor, use others to save!
        ctx.list1 = input_list
        ctx.list2 = matrix_index_list

        ctx.save_for_backward(weight)


        # output = input.mm(weight.t())
        # print("input size:",input.size())
        # print("weight size:",weight.data.size())
        output = torch.matmul(input, weight.t())


        # memory free
        del weight
        del input_list
        del matrix_index_list


        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播计算梯度。

        Args:
            ctx (torch.autograd.function.BackwardCFunction): 上下文对象，存储前向传播的中间信息。
            grad_output (torch.Tensor): 输出的梯度，形状与前向输出一致。

        Returns:
            tuple: 输入、selected_weight、matrix_index_list、weight 的梯度（后两者无梯度返回 None）。
        """
        weight,  = ctx.saved_tensors
        input_list = ctx.list1
        matrix_index_list = ctx.list2

        # Pytorch use C++ engine to check whether gradient has matched dimenstion or not
        grad_weight = torch.empty(len(input_list) * Block_dimension, Block_dimension,dtype=grad_output.dtype,
                                  device=grad_output.device)
        for i in range(len(input_list)):
            index = matrix_index_list[i]

            # print("index:", index)
            # print("grad_output_dim:", grad_output.size())
            # tmp = grad_output.permute(0, 2, 1)[:, index[0] * Block_dimension: index[0] * Block_dimension + Block_dimension, :]
            # print("tmp size", tmp.size())
            # print("input list[i]", input_list[i].size())
            # tmp1 = torch.matmul(tmp, input_list[i])
            # grad_weight[i * Block_dimension: i * Block_dimension + Block_dimension, :] = torch.sum(tmp1, dim=0)

            grad_weight[i * Block_dimension: i * Block_dimension + Block_dimension, :] = torch.sum(torch.matmul(grad_output.permute(0, 2, 1)[:, index[0] * Block_dimension: index[0] * Block_dimension + Block_dimension, :], input_list[i]), dim=0)

        grad_input = torch.matmul(grad_output, weight)

        # memory free
        del weight
        del input_list
        del matrix_index_list

        return grad_input, grad_weight, None, None




def select_submatrix_based_on_grads(grads, n=660, selection_strategy = "no_restriction", calculate_strategy = "mean_abs", model = "yahma/llama-13b-hf"):
    """
    根据梯度信息选择子矩阵。

    Args:
        grads (dict): 每个 MLP 线性权重矩阵的梯度信息，键为模块信息，值为梯度张量。
        n (int, optional): 要选择的子矩阵数量，默认为 660。
        selection_strategy (str, optional): 子矩阵选择策略，默认为 "no_restriction"。
        calculate_strategy (str, optional): 计算策略，默认为 "mean_abs"。
        model (str, optional): 模型名称，默认为 "yahma/llama-13b-hf"。

    Returns:
        dict: 选中的子矩阵及其对应的信息，键为模块信息，值为子矩阵索引列表。
    """
    # Step 1: Calculate absolute value of mean for all grad tensors in every 256x256 block
    if (model == "yahma/llama-13b-hf") or (model == "NousResearch/Llama-2-13b-hf"):
        model="llama-13b"
        Block_dimension = 256
        large_d = 54
        small_d = 20
    # elif model == "yahma/llama-7b-hf":
    elif (model == "NousResearch/Llama-2-7b-hf") or (model == "meta-llama/Llama-2-7b-hf")  \
        or (model == "yahma/llama-7b-hf") or (model == "meta-llama/Llama-2-7b-chat-hf") or (model == "/data/ydh/models/Llama-2-7b-hf"):
        model="llama-2-7b"
        Block_dimension = 256
        large_d = 43
        small_d = 16
    elif (model == "NousResearch/Meta-Llama-3-8B") or (model == "meta-llama/Meta-Llama-3-8B"):
        model="llama-3-8b"
        Block_dimension = 256
        large_d = 56
        small_d = 16

    block_means = {}
    for key, grad in grads.items():
        # Reshape the grad tensor into 256x256 blocks
        if key[0] == 'gate_proj' or key[0] == 'up_proj':
            # print(key[0], grad.size())
            print(f"gate_proj and up_proj dimension check:{key[0]}, {grad.size()}")

            reshaped_grad = grad.reshape(large_d, Block_dimension, small_d, Block_dimension)

        elif key[0] == 'down_proj':
            # print(key[0], grad.size())
            print(f"down_proj dimension check:{key[0]}, {grad.size()}")

            reshaped_grad = grad.reshape(small_d, Block_dimension, large_d, Block_dimension)


        elif key[0] == 'q_proj' or key[0] == 'k_proj' or key[0] == 'v_proj':
            print(f"qkv dimension check:{key[0]}, {grad.size()}")
            if (model == "llama-3-8b") and (key[0] == 'k_proj' or key[0] == 'v_proj'):
                small_d_ = 4
                reshaped_grad = grad.reshape(small_d_, Block_dimension, small_d, Block_dimension)
            else:
                reshaped_grad = grad.reshape(small_d, Block_dimension, small_d, Block_dimension)

    # print("tensor shape:", reshaped_grad.shape)
        if calculate_strategy == 'mean_abs':
            block_means[key] = mean_abs(reshaped_grad)
        elif calculate_strategy == 'abs_mean':
            block_means[key] = abs_mean_(reshaped_grad)
        elif calculate_strategy == 'L1':
            block_means[key] = L1_norm(reshaped_grad)
        elif calculate_strategy == 'L2':
            block_means[key] = L2_norm(reshaped_grad)



    # for each linear layer, select certain number of sub-matrix, normal distributed selection
    if selection_strategy == "norm_dist":
        # Step 2: Rank all the blocks in all grad tensors using the abs.mean() value
        ranked_blocks = defaultdict(list)

        for key, block_mean in block_means.items():
            indices = torch.argsort(block_mean.view(-1), descending=True)
            # print("===================================================")
            # print("indices", indices)
            top_indices = indices[:n]
            for idx in top_indices:
                # may need to consider int memory cost in the future
                row = idx // block_mean.shape[1]
                col = idx % block_mean.shape[1]
                ranked_blocks[key].append((row.item(), col.item()))
        del indices
        del top_indices
        del key
        del block_mean
        # Step 3: Return the selected blocks and their corresponding information
        return ranked_blocks

    else:
        # Step 2: Use a min-heap to maintain top n blocks efficiently
        top_blocks = []
        for key, block_mean in block_means.items():
            for i in range(block_mean.shape[0]):
                for j in range(block_mean.shape[1]):
                    abs_mean = block_mean[i, j].item()
                    if len(top_blocks) < n:
                        heapq.heappush(top_blocks, (abs_mean, (key, i, j)))
                    else:
                        heapq.heappushpop(top_blocks, (abs_mean, (key, i, j)))

        # print("===================================================")
        # print("top_blocks", top_blocks)

        # Step 3: Return the selected top n blocks and their corresponding information
        top_blocks.sort(reverse=True)  # Sort the top_blocks in descending order
        ranked_blocks = defaultdict(list)

        # selected_blocks = [(info, row, col, mean) for mean, (info, row, col) in top_blocks]


        # print("===================================================")
        # print("selected_blocks", selected_blocks)
        # for (info, row, col, mean) in selected_blocks:
        for mean, (info, row, col) in top_blocks:
            ranked_blocks[info].append((row, col))

        del top_blocks
        del mean
        del info
        del key
        del block_mean
        return ranked_blocks


def mean_abs(grad_tensor):
    print(f"use mean()abs() as calculation strategy")
    return grad_tensor.mean(dim=(1, 3)).abs()

def abs_mean_(grad_tensor):
    print(f"use abs()mean() as calculation strategy")
    return grad_tensor.abs().mean(dim=(1, 3))

def L1_norm(grad_tensor):
    print(f"use L1 norm as calculation strategy")

    return grad_tensor.abs().sum(dim=(1, 3))

def L2_norm(grad_tensor):
    print(f"use L2 norm as calculation strategy")
    return torch.sqrt(torch.sum(grad_tensor.abs() ** 2, dim=(1, 3)))





# Source Code from DeepSpeed Examples official website
# Please refer to https://github.com/microsoft/DeepSpeedExamples/blob/75df1d7250452bcc7c065797a95c982bc8caab0b/applications/DeepSpeed-Chat/dschat/utils/utils.py#L210
def get_optimizer_sparse_grouped_parameters(
    model,
    weight_decay,
    smt_lr,
    lora_lr=5e-4,
    no_decay_name_list=[
        "bias", "layer_norm.weight", "layernorm.weight", "norm.weight",
        "ln_f.weight"
    ],
    lora_name_list=["lora_right_weight", "lora_left_weight"],
):


    print(f"================ PRINT PARAM NAME [0]=======================")

    for name, param in model.named_parameters():
        if (not any(nd in name.lower() for nd in no_decay_name_list)
                and param.requires_grad and not any(nd in name.lower() for nd in lora_name_list)):
            print(f"name0:{name}")

    print(f"================ PRINT PARAM NAME [1]=======================")
    for n, p in model.named_parameters():
        if (not any(nd in n.lower() for nd in no_decay_name_list)
                and p.requires_grad and any(nd in n.lower() for nd in lora_name_list)):
            print(f"name1:{n}")



    print(f"================ PRINT PARAM NAME [2]=======================")
    for n, p in model.named_parameters():
        if (any(nd in n.lower() for nd in no_decay_name_list) and p.requires_grad):
            print(f"name2:{n}")



    optimizer_grouped_parameters = [
        {
            "params": #tmp
            [
                p for n, p in model.named_parameters()
                if (not any(nd in n.lower() for nd in no_decay_name_list)
                    and p.requires_grad and not any(nd in n.lower() for nd in lora_name_list))
            ]
            ,
            "weight_decay":
            weight_decay,
            "lr":
            smt_lr
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n.lower() for nd in no_decay_name_list)
                    and p.requires_grad and any(nd in n.lower() for nd in lora_name_list))
            ],
            "weight_decay":
            weight_decay,
            "lr":
            lora_lr
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n.lower()
                        for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":
            0.0,
        },
    ]

    non_empty_groups = []
    for group in optimizer_grouped_parameters:
        if group["params"]:
            non_empty_groups.append(group)

    print(f"group parameters: {non_empty_groups}")

    return non_empty_groups #, sorted_selected_submatrix

