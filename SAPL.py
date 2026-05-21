import torch
import torch_scatter
import torchsparse
import torchsparse.nn.functional
from matplotlib import pyplot as plt
from torch import nn
from utils_source_free.general_imports import * # Assuming this contains necessary imports like networks, dataloaders, etc.
from tqdm import tqdm
import copy
import argparse
import random
import numpy as np
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def torch_unique(x):
    unique, inverse, counts = torch.unique(x, return_inverse=True, return_counts=True)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
    inds = torch_scatter.scatter_min(perm, inverse, dim=0)[0]
    return unique, inds, inverse, counts
def get_point_in_voxel(raw_coord, voxel_size=1.0):
    """ Voxelizes the point cloud and returns basic mappings. """
    voxel_grid = (raw_coord / voxel_size).int()
    hash_tensor = torch.cat((voxel_grid, torch.zeros((voxel_grid.shape[0], 1), device=voxel_grid.device)), dim=1).int()
    pc_hash = torchsparse.nn.functional.sphash(hash_tensor)
    sparse_hash, voxel_idx, inverse, voxel_point_counts = torch_unique(pc_hash)
    return sparse_hash, voxel_idx, inverse, voxel_point_counts
def get_vgi_weights(coords, preds_logits, voxel_size=1.0, alpha=1.0):
    """ Calculates per-point reliability weights using a robust bincount method. """
    _, _, inverse, voxel_point_counts = get_point_in_voxel(coords, voxel_size)
    if inverse.shape[0] == 0:
        return torch.tensor([], device=coords.device)
    num_voxels, num_classes = voxel_point_counts.shape[0], preds_logits.shape[1]
    preds_label = preds_logits.max(dim=-1).indices
    flat_indices = inverse * num_classes + preds_label.long()
    category_counts_flat = torch.bincount(flat_indices, minlength=num_voxels * num_classes)
    category_counts_per_voxel = category_counts_flat.view(num_voxels, num_classes).float()
    class_num_probability = category_counts_per_voxel / (voxel_point_counts[:, None] + 1e-6)
    voxel_impurity = 1 - torch.sum(torch.pow(class_num_probability, 2), dim=1)
    voxel_impurity[voxel_point_counts == 1] = 1.0
    voxel_weights = torch.exp(-1 * alpha * voxel_impurity)
    return voxel_weights[inverse]
def make_a_deepcopy(net, logging):
    """
    对输入的网络进行深拷贝，并将其设置为评估模式。

    :param net: 待拷贝的网络模型
    :param logging: 日志记录器，用于记录信息
    :return: 深拷贝后的网络模型，且已设置为评估模式
    """
    # 对输入的网络进行深拷贝
    net_copy = copy.deepcopy(net)
    # 遍历深拷贝后网络的所有模块
    for l_name, l_module in net_copy.named_modules():
        # 如果模块是批量归一化层，则将其设置为评估模式
        if isinstance(l_module, torch.nn.modules.batchnorm._BatchNorm):
            l_module.eval()

    # 将深拷贝后的网络整体设置为评估模式
    net_copy.eval()
    return net_copy
def validation_performance_pseudo_labels(net, config, test_loader, device, cls_thresh_cuda,
                                         list_ignore_classes=[0], running_conf=None, THRESHOLD_BETA=None, mapping_information=None):
    """
    该函数用于报告伪标签的性能。

    :param net: 用于验证的网络模型
    :param config: 配置字典，包含各种参数设置
    :param test_loader: 测试数据加载器
    :param device: 计算设备，如 'cuda' 或 'cpu'
    :param cls_thresh_cuda: 类别阈值，在设备上的张量
    :param list_ignore_classes: 要忽略的类别列表，默认为 [0]
    :param running_conf: 运行置信度，默认为 None
    :param THRESHOLD_BETA: 阈值 beta，默认为 None
    :return: 包含验证结果的字典
    """
    # 将网络设置为评估模式
    net.eval()
    # 初始化分割头的误差
    error_seg_head = 0
    # 初始化伪标签率的总和
    pseudo_label_rate_sum = 0.0
    # 初始化忽略伪标签率的总和
    ignore_pseudo_label_rate_sum = 0.0

    # 初始化分割头的混淆矩阵
    cm_seg_head = np.zeros((config["nb_classes_inference"], config["nb_classes_inference"]))
    # 加载源类映射信息
    if mapping_information is None:
        mapping_information = sf_class_mapping_loader(source_dataset=config["source_dataset_name"],
                                                  target_dataset=config["target_dataset_name"])
    # 生成求和矩阵
    summation_matrix = summation_matrix_generator(mapping_information)
    # 将求和矩阵移动到指定设备上
    summation_matrix = summation_matrix.to(device)
    # 上下文管理器，禁用梯度计算
    with torch.no_grad():
        # 初始化迭代计数器
        count_iter = 0

        # 遍历测试数据加载器中的数据
        for data in test_loader:

            # 将数据移动到指定设备上
            data = dict_to_device(data, device)
            # 前向传播，获取分割输出
            _, output_seg, _ = net.forward_mapped_learned(data)  # SOURCE data

            # 根据配置决定是否进行映射
            if config["parameter"]["pl_no_mapping"]:
                output_merged = output_seg[:, :, 0]
            else:
                output_merged = output_seg[:, :, 0] @ summation_matrix

            # 计算伪标签，对输出进行 softmax 操作
            output = F.softmax(output_merged, dim=1)

            ###DT-ST setting
            # 生成阈值化的标签
            thresolded_label, _, _ = pseudo_labels_probs(output, running_conf, THRESHOLD_BETA)
            # 将阈值化的标签移动到 CPU 并转换为 numpy 数组
            thresolded_label = thresolded_label.detach().cpu().numpy()
            # else:
            #     thresolded_label = label_selection(cls_thresh_cuda, output).cpu().numpy()

            # 获取当前的预测结果，并进行预测转换
            output_seg_np = prediction_changer(output_seg.cpu().detach(),
                                               mapping_information)  # Only predicting on available classes
            # 获取目标分割标签，并转换为整数类型
            target_seg_np = data["y"].cpu().numpy().astype(int)

            # 仅评估不是映射到 0 的伪标签
            mask_pseudo_labels = thresolded_label != 0
            # 获取伪标签对应的预测结果
            output_seg_np_pseudo_label = output_seg_np[mask_pseudo_labels]
            # 获取伪标签对应的目标标签
            target_seg_np_pseudo_label = target_seg_np[mask_pseudo_labels]

            # 计算当前批次的伪标签率
            pseudo_label_rate_batch = 1 - (np.sum(mask_pseudo_labels) / mask_pseudo_labels.shape[0])
            # 累加伪标签率
            pseudo_label_rate_sum += pseudo_label_rate_batch

            # 获取伪标签对应的真实标签
            true_labels_pseudo_labels = data["y"].detach()[mask_pseudo_labels]
            # 生成忽略点的掩码
            mask_ignore_points_in_pseudo_labels = true_labels_pseudo_labels == 0
            # 计算当前批次忽略点的损失率
            rate_ignore_points_loss_batch = np.sum(mask_ignore_points_in_pseudo_labels.detach().cpu().numpy()) / np.sum(
                mask_pseudo_labels)
            # 累加忽略点的损失率
            ignore_pseudo_label_rate_sum += rate_ignore_points_loss_batch

            # 计算当前批次的混淆矩阵
            cm_seg_head_ = confusion_matrix(output_seg_np_pseudo_label.ravel(), target_seg_np_pseudo_label.ravel(),
                                            labels=list(range(config["nb_classes_inference"])))
            # 累加混淆矩阵
            cm_seg_head += cm_seg_head_

            # 迭代计数器加 1
            count_iter += 1
            # 每 10 次迭代清空一次 CUDA 缓存
            if count_iter % 10 == 0:
                torch.cuda.empty_cache()
        # 计算训练分割头的逐点分数
        # test_seg_head_oa = metrics.stats_overall_accuracy(cm_seg_head, ignore_list=list_ignore_classes)
        # 计算分割头的平均准确率和每类准确率
        test_seg_head_maa, accuracy_per_class = metrics.stats_accuracy_per_class(cm_seg_head,
                                                                                 ignore_list=list_ignore_classes)  # First return value is the mean IoU
        # 计算分割头的平均交并比和每类交并比
        test_seg_head_miou, seg_iou_per_class = metrics.stats_iou_per_class(cm_seg_head,
                                                                            ignore_list=list_ignore_classes)  # First return value is the mean IoU
        # 计算分割头的损失
        test_seg_head_loss = error_seg_head / cm_seg_head.sum()

    # 定义返回结果的字典
    return_data = {
        "test_seg_head_miou": test_seg_head_miou,
        "test_seg_head_maa": test_seg_head_maa,
        "test_seg_head_loss": test_seg_head_loss,
        "seg_iou_per_class": seg_iou_per_class,
        "accuracy_per_class": accuracy_per_class,
        "cm_seg_head": cm_seg_head,
        "pseudo_label_rate": pseudo_label_rate_sum / count_iter,
        "ignore_points_rate": ignore_pseudo_label_rate_sum / count_iter
    }
    return return_data
def pseudo_label(net_pseudo_label, config, target_train_loader, device, writer, names_list, i, running_conf,
                 THRESHOLD_BETA, mapping_information=None):
    """
    生成伪标签并记录相关指标。

    :param net_pseudo_label: 用于生成伪标签的网络模型
    :param config: 配置字典，包含各种参数设置
    :param target_train_loader: 目标训练数据加载器
    :param device: 计算设备，如 'cuda' 或 'cpu'
    :param writer: 用于记录指标的写入器
    :param names_list: 类别名称列表
    :param i: 当前迭代次数
    :param running_conf: 运行置信度
    :param THRESHOLD_BETA: 阈值 beta
    :return: 类别阈值，在设备上的张量
    """
    print("Running pseudo_label()...")
    # 不需要重新校准阈值，初始化类别阈值
    cls_thresh = np.ones(config["nb_classes"], dtype=np.float32)

    # 将类别阈值移动到指定设备上
    cls_thresh_cuda = torch.from_numpy(cls_thresh).to(device)

    # 调用验证函数，获取伪标签的验证结果
    return_data_pl = validation_performance_pseudo_labels(net_pseudo_label, config, target_train_loader, device,
                                                          cls_thresh_cuda, [0], running_conf, THRESHOLD_BETA, mapping_information)

    # 记录伪标签的平均交并比
    logging.info(f"Pseudo Label mIoU: {return_data_pl['test_seg_head_miou']}")
    # 将伪标签的平均交并比写入 TensorBoard
    writer.add_scalar(f"pseudo_label.seg_mIou", return_data_pl['test_seg_head_miou'], i)
    # 记录伪标签的每类交并比
    logging.info(f"Pseudo Label Per class {return_data_pl['seg_iou_per_class']}")
    # 将每类交并比写入 TensorBoard
    for q in range(len(names_list)):
        writer.add_scalar(f"pseudo_label.seg_Iou_{names_list[q]}", return_data_pl['seg_iou_per_class'][q], i)
    # 记录伪标签率
    logging.info(f"Pseudo Label rate: {return_data_pl['pseudo_label_rate']}")
    # 将伪标签率写入 TensorBoard
    writer.add_scalar(f"pseudo_label.pseudo_label_rate", return_data_pl['pseudo_label_rate'], i)

    # 记录伪标签中忽略类别的比率
    logging.info(f"Ignore class rates in pseudo label {return_data_pl['ignore_points_rate']}")
    # 将伪标签中忽略类别的比率写入 TensorBoard
    writer.add_scalar(f"pseudo_label.evaluation.pseudo_ignore_points_rate", return_data_pl['ignore_points_rate'], i)

    return cls_thresh_cuda
def pseudo_labels_probs(probs, running_conf, THRESHOLD_BETA, RUN_CONF_UPPER=0.80, ignore_augm=None, discount=True):

    ### From https://github.com/DZhaoXd/DT-ST/blob/main/train_TCR_DTU.py#L94
    """Consider top % pixel w.r.t. each image"""
    ###We consider the whole batch

    # 置信度上限
    RUN_CONF_UPPER = RUN_CONF_UPPER
    # 置信度下限
    RUN_CONF_LOWER = 0.20

    # 获取概率张量的批次大小和类别数
    N, C = probs.size()
    # 获取每个样本的最大置信度和对应的索引
    max_conf, max_idx = probs.max(1, keepdim=True)  # B,1,H,W, take per example the maximum

    # 初始化峰值概率张量
    probs_peaks = torch.zeros_like(probs)
    # 将最大置信度写入峰值概率张量
    probs_peaks.scatter_(1, max_idx, max_conf)  # B,C,H,W #Write into the zero array the maximum per example

    # 获取每个类别的最大峰值概率
    top_peaks, _ = probs_peaks.view(N, C).max(
        0)  # N,C #Get the top peaks per class for the complete batch --> we assume bs=1

    # 乘以置信度上限
    top_peaks *= RUN_CONF_UPPER

    if discount:
        # 对长尾类别的阈值进行折扣
        top_peaks *= (1. - torch.exp(- running_conf / THRESHOLD_BETA))

    # 对峰值概率进行下限裁剪
    top_peaks.clamp_(RUN_CONF_LOWER)  # in-place --> set to a minimal threshold of 20
    # 生成大于峰值概率的掩码
    probs_peaks.gt_(top_peaks.view(1, C))

    # 如果低于折扣后的峰值，则忽略
    ignore = probs_peaks.sum(1, keepdim=True) != 1

    # 阈值化最置信的像素，生成伪标签
    pseudo_labels = max_idx.clone()
    pseudo_labels[ignore] = 0
    pseudo_labels = pseudo_labels.squeeze(1)

    return pseudo_labels, max_conf, max_idx
# refer to https://github.com/visinf/da-sac
def update_running_conf(probs, running_conf, THRESHOLD_BETA, tolerance=1e-8):
    """
    维护移动类先验。

    :param probs: 概率张量
    :param running_conf: 运行置信度
    :param THRESHOLD_BETA: 阈值 beta
    :param tolerance: 容差，默认为 1e-8
    :return: 更新后的运行置信度
    """
    # 统计动量
    STAT_MOMENTUM = 0.9

    # 获取概率张量的批次大小和类别数
    N, C = probs.size()
    # 计算概率的平均值
    probs_avg = probs.mean(0).view(C, -1).mean(-1)
    # 找到需要更新的索引
    update_index = probs_avg > tolerance
    # 找到新的索引
    new_index = update_index & (running_conf == THRESHOLD_BETA)
    # 更新新记录的运行置信度
    running_conf[new_index] = probs_avg[new_index]

    # 对其余部分使用移动平均
    running_conf *= STAT_MOMENTUM
    running_conf += (1 - STAT_MOMENTUM) * probs_avg
    return running_conf
def entropy(p, prob=True, mean=True):
    """
    计算熵。

    :param p: 输入张量
    :param prob: 是否进行 softmax 操作，默认为 True
    :param mean: 是否返回均值，默认为 True
    :return: 熵值
    """
    if prob:
        # 对输入进行 softmax 操作
        p = F.softmax(p, dim=1)
    # 计算熵
    en = -torch.sum(p * torch.log(p + 1e-5), 1)
    if mean:
        return torch.mean(en)
    else:
        return en
class WeightEMA(object):
    """
    指数移动平均类，用于更新模型参数。
    """

    def __init__(self, params, src_params, alpha):
        """
        初始化 WeightEMA 类。

        :param params: 待更新的参数列表
        :param src_params: 源参数列表
        :param alpha: 指数移动平均的系数
        """
        self.params = list(params)
        self.src_params = list(src_params)
        self.alpha = alpha

    def step(self):
        """
        执行指数移动平均更新。
        """
        one_minus_alpha = 1.0 - self.alpha
        for p, src_p in zip(self.params, self.src_params):
            p.data.mul_(self.alpha)
            p.data.add_(src_p.data * one_minus_alpha)
def evel_stu(config, net, stu_eval_list, device, dict_with_mapping, alpha):
    """
    评估学生模型。

    :param config: 配置字典，包含各种参数设置
    :param net: 学生模型
    :param stu_eval_list: 学生评估列表
    :param device: 计算设备，如 'cuda' 或 'cpu'
    :param dict_with_mapping: 包含映射信息的字典
    :param alpha: 系数
    :return: 评估结果列表
    """
    # 将网络设置为评估模式
    net.eval()

    # 初始化评估结果列表
    eval_result = []

    # 上下文管理器，禁用梯度计算
    with torch.no_grad():
        # 遍历学生评估列表
        for i, (target_data, permute_index) in enumerate(stu_eval_list):
            # 将数据移动到指定设备上
            target_data = dict_to_device(target_data, device)
            # 前向传播，获取分割输出
            _, output, _ = net.forward_mapped_learned(target_data)
            # 对输出进行 softmax 操作
            output = F.softmax(output, dim=1)[:, :, 0]
            # 获取随机排列的索引
            pred1_rand = permute_index
            # 选择的点数
            select_point = 100
            # 对选择的输出进行归一化
            pred1 = F.normalize(output[pred1_rand[:select_point]])
            # 计算熵
            pred1_en = entropy(torch.matmul(pred1, pred1.t()) * 20)
            # 将评估结果添加到列表中
            eval_result.append(pred1_en.item())

    # 将网络设置为训练模式
    net.train()
    # 遍历网络的所有模块
    for l_name, l_module in net.named_modules():
        # 如果模块是批量归一化层，则将其设置为评估模式
        if isinstance(l_module, torch.nn.modules.batchnorm._BatchNorm):
            l_module.eval()

    return eval_result
# ======================================================================================================
class SAPL_Module(nn.Module):
    def __init__(self, num_classes, model, alpha=0.9, gsp_sigma=10.0):

        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.min_feat_points = 1
        self.max_experts = 6  # Initialize a pool of 5 experts
        self.gsp_sigma = gsp_sigma
        # self.std_thresholds = std_thresholds

        initial_prototypes = copy.deepcopy(model.dual_seg_head.weight.data.squeeze(-1).T)
        self.expert_prototypes = nn.ParameterList(
            [nn.Parameter(copy.deepcopy(initial_prototypes)) for _ in range(self.max_experts)]
        )
        for p in self.expert_prototypes:
            p.data = F.normalize(p.data, p=2, dim=0)

    def _kmeans_1d_gpu(self, data, K, max_iters=10):
        """ [MODIFIED] Returns both RSS and the final centers (centroids). """
        if data.numel() == 0:
            return 0.0, torch.zeros(K, device=data.device)
        """ A simple and fast 1D K-Means implementation on GPU. """
        centers = torch.linspace(data.min(), data.max(), K, device=data.device)
        for _ in range(max_iters):
            diffs = torch.abs(data.unsqueeze(1) - centers.unsqueeze(0))
            assignments = torch.argmin(diffs, dim=1)
            new_centers = torch_scatter.scatter_mean(data, assignments, dim=0, dim_size=K)
            # Handle empty clusters by re-initializing them to a random value
            is_nan = torch.isnan(new_centers)
            if torch.any(is_nan):
                new_centers[is_nan] = torch.rand(is_nan.sum(), device=data.device) * data.max()
            if torch.allclose(centers, new_centers, atol=1e-4):
                break
            centers = new_centers
        cluster_centers = centers[assignments]
        rss = torch.sum((data - cluster_centers) ** 2)
        return rss, centers

    def determine_num_experts_bic(self, partition_values, k_candidates=None):
        if k_candidates is None:
            k_candidates = [2, 3, 4, 5, 6]
        if partition_values.shape[0] < 20: return 2, torch.tensor([0.0, 1.0], device=partition_values.device)
        n = partition_values.shape[0]
        log_n = torch.log(torch.tensor(n, device=partition_values.device))

        bics, all_centers = [], []

        for k in k_candidates:
            if n < k: continue
            rss, centers = self._kmeans_1d_gpu(partition_values, K=k)
            bic = n * torch.log(rss / n + 1e-6) + k * log_n
            bics.append(bic)
            all_centers.append(centers)

        if not bics: return 2, torch.tensor([partition_values.min(), partition_values.max()],
                                            device=partition_values.device)

        best_idx = torch.argmin(torch.stack(bics))
        best_k = k_candidates[best_idx]
        best_centers = all_centers[best_idx]

        return best_k, best_centers

    # def determine_num_experts(self, points_coords):
    #     """ Determines the number of experts to use based on scene depth complexity. """
    #     if points_coords.shape[0] < 2:
    #         return 2  # Default for very sparse clouds
    #     distances = torch.sqrt(points_coords[:, 0] ** 2 + points_coords[:, 1] ** 2 + points_coords[:, 2] ** 2)
    #     depth_std = torch.std(distances)
    #     print(f"depth_std:{depth_std}")
    #     if depth_std < self.std_thresholds[0]:
    #         return 2  # Simple scene
    #     elif depth_std < self.std_thresholds[1]:
    #         return 3  # Medium complexity scene
    #     else:
    #         return 4  # Complex scene, use more experts

    # def _get_soft_partition_weights(self, points_coords, num_active_experts):
    #     """ Calculates Gaussian weights for a DYNAMIC number of experts. """
    #     distances = torch.sqrt(points_coords[:, 0] ** 2 + points_coords[:, 1] ** 2)
    #     # distances = torch.sqrt(points_coords[:, 0] ** 2 + points_coords[:, 1] ** 2 + points_coords[:, 2] ** 2)
    #     max_dist = torch.max(distances).cpu().item() if points_coords.shape[0] > 0 else 0.0
    #     anchors = torch.linspace(0.0, max_dist, num_active_experts, device=distances.device)
    #     dist_matrix = distances.unsqueeze(1) - anchors.unsqueeze(0)
    #     weights = torch.exp(-(dist_matrix ** 2) / (2 * self.gsp_sigma ** 2))
    #     return weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

    def _get_soft_partition_weights(self, partition_values, anchors):
        dist_matrix = partition_values.unsqueeze(1) - anchors.unsqueeze(0)
        weights = torch.exp(-(dist_matrix ** 2) / (2 * self.gsp_sigma ** 2))
        return weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
    def vgi_ent_filter(self, preds_logits, predictions, point_coords, voxel_size=0.5, percent=0.7):

        # _, _, inverse, _, voxel_point_counts = get_point_in_voxel(point_coords, voxel_size)
        _, _, inverse, voxel_point_counts = get_point_in_voxel(point_coords, voxel_size)
        if inverse.shape[0] == 0:
            return torch.tensor([], dtype=torch.long, device=point_coords.device)

        num_voxels = voxel_point_counts.shape[0]
        flat_indices = inverse * self.num_classes + predictions.squeeze(-1).long()
        category_counts_flat = torch.bincount(flat_indices, minlength=num_voxels * self.num_classes)
        category_counts_per_voxel = category_counts_flat.view(num_voxels, self.num_classes).float()

        proportions = category_counts_per_voxel / (voxel_point_counts[:, None] + 1e-6)
        voxel_impurity = 1 - torch.sum(torch.pow(proportions, 2), dim=1)
        voxel_impurity[voxel_point_counts == 1] = 1.0

        point_impurity = voxel_impurity[inverse]
        ent = entropy(preds_logits, prob=True, mean=False).squeeze(-1)
        point_reliability_score = point_impurity * ent

        k = int(point_coords.shape[0] * percent)
        if k == 0:
            return torch.tensor([], dtype=torch.long, device=point_coords.device)
        _, reliable_indices = torch.topk(point_reliability_score, k, largest=False)
        return reliable_indices

    def update_prototypes(self, reliable_features, reliable_labels, reliable_partition_values, reliable_vgi_weights,
                          anchors):
        soft_partition_weights = self._get_soft_partition_weights(reliable_partition_values, anchors)
        num_active_experts = len(anchors)
        for cls_idx in range(self.num_classes):
            class_mask = (reliable_labels == cls_idx).squeeze(-1)
            if class_mask.sum() < self.min_feat_points: continue
            class_features = reliable_features[class_mask]
            class_soft_weights = soft_partition_weights[class_mask]
            class_vgi_weights = reliable_vgi_weights[class_mask].unsqueeze(1)
            combined_weights = class_soft_weights * class_vgi_weights
            for expert_idx in range(num_active_experts):
                expert_specific_weights = combined_weights[:, expert_idx].unsqueeze(1)
                if expert_specific_weights.sum() < 1e-6: continue
                weighted_mean_feature = (class_features * expert_specific_weights).sum(
                    dim=0) / expert_specific_weights.sum()
                current_prototype = self.expert_prototypes[expert_idx][:, cls_idx]
                updated_prototype = self.alpha * current_prototype + (1 - self.alpha) * F.normalize(
                    weighted_mean_feature, p=2, dim=0)
                self.expert_prototypes[expert_idx].data[:, cls_idx] = updated_prototype

    def get_prototype_similarities(self, features, partition_values, anchors):
        soft_weights = self._get_soft_partition_weights(partition_values, anchors)
        num_active_experts = len(anchors)
        norm_features = F.normalize(features, p=2, dim=1)
        active_prototypes = torch.stack(
            [F.normalize(self.expert_prototypes[i], p=2, dim=0) for i in range(num_active_experts)])
        sim_per_expert = torch.einsum('nf,efc->nec', norm_features, active_prototypes)
        weighted_sim = sim_per_expert * soft_weights.unsqueeze(-1)
        return weighted_sim.sum(dim=1)

def main(config_arguments):
    """
    主函数，用于训练和评估模型。
    :param config_arguments: 配置参数字典
    """
    if True:
        # 设置随机种子
        torch.manual_seed(1234)
        random.seed(1234)
        np.random.seed(1234)

        # 启用自动求导异常检测
        torch.autograd.set_detect_anomaly(True)
        # 初始化 TensorBoard 写入器
        writer = SummaryWriter(
            'runs_eccv/{}'.format(f"{config_arguments['tensorboard_folder']}/{config_arguments['name']}"))

        if os.path.isfile(config_arguments["resume_path"]):
            # 如果指定的检查点文件存在，则获取配置文件路径
            print("File exists")
            file_path_config = os.path.join(os.path.dirname(config_arguments["resume_path"]), "config.yaml")
        else:
            # 否则，从指定路径获取配置文件路径
            file_path_config = os.path.join(config_arguments["resume_path"], "config.yaml")

        # 读取配置文件
        config = read_yaml_file(file_path_config)
        if config:
            # 如果配置文件读取成功，打印配置文件路径和配置信息
            print(file_path_config)
            print(f"Loaded Config: {config}")
        else:
            # 如果配置文件读取失败，打印错误信息
            print("Failed to read the config YAML file.")
        # 设置日志级别
        logging.getLogger().setLevel(config["logging"])
        # 将配置参数更新到配置字典中
        config["parameter"] = config_arguments

        ##############################################################################################
        # Selection of the setting that is used

        # 调整配置
        config = config_adapter(config)

        # 设置要忽略的类别
        config["ignore_class"] = 0

        # 加载源类映射信息
        mapping_info = sf_class_mapping_loader(source_dataset=config["source_dataset_name"],
                                               target_dataset=config["target_dataset_name"])
        # 生成求和矩阵
        summation_matrix = summation_matrix_generator(mapping_info)
        ##############################################################################################

        ### Iterate over the additional arguments
        # 遍历配置参数并记录日志
        for k, v in config_arguments.items():
            logging.info(f"{k}: {v}")

        # 记录创建网络的信息
        logging.info("Creating the network")
        logging.info(f"Self-Supervised Setting")
        # 设置训练批次大小
        config['training_batch_size'] = config["parameter"]["batch_size"]
        # 设置测试批次大小
        config["test_batch_size"] = 16
        # 定义保存模型的根目录
        savedir_root = f"ckpts_bn/{config_arguments['name']}"
        # 创建保存模型的根目录
        os.makedirs(savedir_root, exist_ok=True)
        # 设置数据集版本
        config["ns_dataset_version"] = 'v1.0-trainval'
        # 设置网络骨干
        config["network_backbone"] = 'TorchSparseMinkUNet_learned'

        # 初始化名称逆映射字典
        name_shift_inverse = {}

        # 构建名称逆映射字典
        for key, value in name_shift.items():
            name_shift_inverse[value] = key

        # 设置固定头模型的路径
        config["da_fixed_head_path_model"] = config["parameter"]["resume_path"]

        # 定义计算设备
        device = torch.device(config['device'])
        if config["device"] == "cuda":
            # 如果使用 CUDA 设备，启用 cuDNN 基准模式
            torch.backends.cudnn.benchmark = True

        # 获取骨干网络的根目录
        bb_dir_root = get_bbdir_root(config)
        # 创建网络
        latent_size = config["network_latent_size"]
        backbone = config["network_backbone"]
        decoder = {'name': config["network_decoder"], 'k': config['network_decoder_k']}

        # 获取源和目标的输入通道数
        in_channels_source, _, in_channels_target, _ = da_get_inputs(config)
        # 记录源输入通道数
        logging.info("In channels source {}".format(in_channels_source))
        # 记录目标输入通道数
        logging.info("in channels target {}".format(in_channels_target))
        # 记录创建网络的信息
        logging.info("Creating the network")

        def network_function():
            """
            定义创建网络的函数。

            :return: 网络模型
            """
            return networks.Network(
                in_channels=in_channels_source,
                latent_size=latent_size,
                backbone=backbone,
                voxel_size=config["voxel_size"],
                dual_seg_head=config["dual_seg_head"],
                target_in_channels=in_channels_target,
                config=config
            )

        ### Final network
        # 创建最终的网络模型
        net_student = network_function()
        ## ckpt_number = -1 means to load the last ckpt
        if os.path.isfile(bb_dir_root):
            # 如果指定的检查点文件存在，则加载检查点
            ckpt_path = bb_dir_root
            logging.info(f"Failed: Load ckpt from {ckpt_path}")

        # 加载检查点
        checkpoint = torch.load(ckpt_path, map_location=device)

        # Updating the checkpoint
        # 初始化新的检查点字典
        checkpoint_new = {}
        # 遍历检查点的状态字典
        for key in checkpoint["state_dict"].keys():
            if key in name_shift_inverse:
                # 如果键存在于名称逆映射字典中，则更新键名
                checkpoint_new[name_shift_inverse[key]] = checkpoint["state_dict"][key]
            else:
                if "num_batches_tracked" in key or "point_transforms" in key:
                    # 如果键包含特定字符串，则跳过
                    pass
                else:
                    # 否则，直接复制键值对
                    checkpoint_new[key] = checkpoint["state_dict"][key]

        try:
            # 尝试加载更新后的检查点
            net_student.load_state_dict(checkpoint_new)
        except Exception as e:
            # 如果加载失败，打印错误信息并尝试非严格加载
            print(e)
            logging.info(
                f"Loaded parameters do not match exactly net architecture, switching to load_state_dict strict=false")
            net_student.load_state_dict(checkpoint_new, strict=False)

        # 记录网络的参数数量
        logging.info(f"Network -- Number of parameters {count_parameters(net_student)}")
        # 获取目标数据集类
        target_DatasetClass = get_dataset(eval("datasets." + config["target_dataset_name"]))

        # 验证集编号
        val_number = 1  # 1: verifying split, 2 train split, else: test split

        # 打印配置数据集信息
        print(f"config dataset source {config}")
        # 获取数据加载器字典
        dataloader_dict = da_sf_get_dataloader(target_DatasetClass, config, net_student, network_function, val=val_number,
                                               train_shuffle=True, keep_orignal_data=True)

        # 获取目标训练数据加载器
        target_train_loader = dataloader_dict["target_train_loader"]
        # 获取目标测试数据加载器
        target_test_loader = dataloader_dict["target_test_loader"]

        # if config_arguments['run_analysis']:
        #     run_density_analysis_and_plot(config, target_train_loader)
        #     return  # Exit after analysis

        # 创建保存模型的根目录
        os.makedirs(savedir_root, exist_ok=True)
        # 保存配置文件
        save_config_file(eval(str(config)), os.path.join(savedir_root, "config.yaml"))

        # 创建损失层
        loss_layer = torch.nn.BCEWithLogitsLoss()
        # 初始化自监督损失的权重
        weights_ss = torch.ones(config["nb_classes_inference"])
        # 获取要忽略的类别列表
        list_ignore_classes = ignore_selection(config["ignore_idx"])
        # 设置要忽略类别的权重为 0
        for idx_ignore_class in list_ignore_classes:
            weights_ss[idx_ignore_class] = 0
        # 记录要忽略的类别
        logging.info(f"Ignored classes {list_ignore_classes}")
        # 记录不同类别的权重
        logging.info(f"Weights of the different classes {weights_ss}")
        # 将权重移动到指定设备上
        weights_ss = weights_ss.to(device)
        # 创建交叉熵损失层
        # ce_loss_layer = torch.nn.CrossEntropyLoss(weight = weights_ss)
        ce_loss_layer = torch.nn.CrossEntropyLoss(weight=weights_ss, reduction="none")

        ###For all classes, not just the inference one
        # 初始化所有类别的权重
        weights_ss_all = torch.ones(config["nb_classes"])
        # 获取要忽略的类别列表
        list_ignore_classes = ignore_selection(config["ignore_idx"])
        # 设置要忽略类别的权重为 0
        for idx_ignore_class in list_ignore_classes:
            weights_ss_all[idx_ignore_class] = 0
        # 记录要忽略的类别
        logging.info(f"Ignored classes {list_ignore_classes}")
        # 记录所有类别的权重
        logging.info(f"Weights of the different classes for all classes {weights_ss_all}")
        # 将权重移动到指定设备上
        weights_ss_all = weights_ss_all.to(device)
        # 创建交叉熵损失层
        ce_loss_layer_all = torch.nn.CrossEntropyLoss(weight=weights_ss_all, reduction="none")
        # ce_loss_layer_all = torch.nn.CrossEntropyLoss(weight = weights_ss_all)

        # 将网络设置为评估模式
        net_student.eval()
        # 将网络移动到指定设备上
        net_student.to(device)

        # 初始化包含映射信息的字典
        dict_with_mapping = {}
        # 初始化系数
        alpha = None

        # 初始化要更新的参数列表
        list_parameter_to_update = []
        # 初始化其他参数列表
        list_parameter_others = []  # 2nd section of selected parameters, e.g. if scaling LL and Backbone differently

        # 配置冻结模型，并获取要更新的参数列表
        net_student, list_parameter_to_update, list_parameter_others = \
            configure_freeze_models(net_student, config, list_parameter_to_update, list_parameter_others)

        # 遍历网络的所有模块
        for l_name, l_module in net_student.named_modules():
            # 如果模块是批量归一化层，则将其设置为评估模式
            if isinstance(l_module, torch.nn.modules.batchnorm._BatchNorm):
                l_module.eval()

        # 初始化类先验
        class_prior = np.zeros((1))
        # 获取类先验和类别名称列表
        class_prior, names_list = class_prior_class_names(config, logging)

        # Obtain the class priors from target training set
        # 初始化系数
        alpha = 0
        # 将系数转换为张量并移动到指定设备上
        alpha_tensor = (torch.ones(1) * alpha).to(device)

        # Weight for entropy loss
        # 初始化熵损失的权重
        ent_weigth = np.array(config["parameter"]["ent_weigth"]).astype(np.float64)
        # 将熵损失的权重转换为张量并移动到指定设备上
        ent_weigth = torch.from_numpy(ent_weigth).type(torch.FloatTensor).to(device)

        # 初始化伪标签损失的权重
        pl_weigth = np.array(config["parameter"]["pl_weigth"]).astype(np.float64)
        # 将伪标签损失的权重转换为张量并移动到指定设备上
        pl_weigth = torch.from_numpy(pl_weigth).type(torch.FloatTensor).to(device)

        # 将求和矩阵移动到指定设备上
        summation_matrix = summation_matrix.to(device)
        # 将类先验转换为张量并移动到指定设备上
        class_prior = torch.from_numpy(class_prior).to(device)

        ### From https://github.com/DZhaoXd/DT-ST/blob/main/train_TCR_DTU.py#L325
        ###### confident  init
        # default param in SAC (https://github.com/visinf/da-sac)
        # 初始化阈值 beta
        THRESHOLD_BETA = 0.001
        # 初始化运行置信度
        running_conf = torch.zeros(config["nb_classes"]).cuda()
        # 将运行置信度填充为阈值 beta
        running_conf.fill_(THRESHOLD_BETA)

        ###### Dynamic teacher init
        # 初始化学生评估列表
        stu_eval_list = []
        # 初始化学生分数缓冲区
        stu_score_buffer = []
        # 初始化结果字典
        res_dict = {'stu_ori': [], 'stu_now': [], 'update_iter': []}

        # Initialisation of the Pseudo-Label backbone
        # 初始化伪标签骨干网络
        net_teacher = make_a_deepcopy(net_student, logging)
        if config["parameter"]["ema_teacher"]:
            # 如果使用 EMA 教师模型，则初始化 EMA 优化器
            net_his_optimizer = WeightEMA(
                list(net_teacher.parameters()),
                list(net_student.parameters()),
                alpha=0.99)

        # 将网络移动到指定设备上
        net_student.to(device)
        if config["parameter"]["finetune"] and config["parameter"]["fintune_setting"] == "classic":
            # 如果进行微调且设置为经典模式，则初始化优化器
            logging.info("Classifier get updated with 10X higher LR than backbone.")
            optimizer = torch.optim.AdamW([
                {"params": list_parameter_to_update, "lr": config["parameter"]["learning_rate"]},
                {"params": list_parameter_others, "lr": config["parameter"]["learning_rate"] / 10.0}
            ])  # Backbone is updated with a 10x smaller learning rate
        else:
            # 否则，初始化优化器
            optimizer = torch.optim.AdamW([{"params": list_parameter_to_update}], config["parameter"]["learning_rate"])
        if config["parameter"]["lr_scheduler"]:
            # 如果使用学习率调度器，则初始化学习率调度器
            scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=config["parameter"]["nb_iterations"],
                                                              power=0.9, last_epoch=-1, verbose=False)

    sapl_module = SAPL_Module(num_classes=config["nb_classes"], model=net_student)
    # 获取目标训练数据加载器的迭代器
    train_iter_trg = enumerate(target_train_loader)
    best_miou = 0
    # 开始训练迭代
    for i in tqdm(range(config["parameter"]["nb_iterations"]), desc="Training"):

        if i == 0:pass
        elif i % config["parameter"]["val_intervall"] == 0:
            # 每间隔一定迭代次数进行验证
            logging.info(i)

            # 进行非预映射的验证
            return_data_val_target_mapped = \
                validation_non_premap(net_student, config, target_test_loader, epoch=0, disable_log=False, device=device,
                                      list_ignore_classes=[0], mapping_information=mapping_info)
            # 记录验证的平均交并比
            logging.info(f"mIoU: {return_data_val_target_mapped['test_seg_head_miou']}")
            # 将验证的平均交并比写入 TensorBoard
            writer.add_scalar(f"validation.seg_mIou", return_data_val_target_mapped['test_seg_head_miou'], i)
            # 记录验证的每类交并比
            logging.info(f"Per class {return_data_val_target_mapped['seg_iou_per_class']}")
            # 将每类交并比写入 TensorBoard
            for q in range(len(names_list)):
                writer.add_scalar(f"validation.seg_Iou_{names_list[q]}",
                                  return_data_val_target_mapped['seg_iou_per_class'][q], i)

            # After validation, set again the BN to the defined setting
            # 验证后，将批量归一化层设置为定义的模式
            for l_name, l_module in net_student.named_modules():
                if isinstance(l_module, torch.nn.modules.batchnorm._BatchNorm):
                    l_module.eval()

            # 记录下最高的miou
            if return_data_val_target_mapped['test_seg_head_miou'] > best_miou:
                best_miou = return_data_val_target_mapped['test_seg_head_miou']
                # 最高miou对应的每个类别iou
                best_miou_per_class = return_data_val_target_mapped['seg_iou_per_class']
                best_epoch = i
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print(f"setting:{config['parameter']['setting']}")
                print(f'better miou:{best_miou},epoch:{best_epoch}')
                with open(os.path.join(savedir_root, f"best_miou.csv"), "w") as f:
                    f.write(f"best_miou,{best_miou}\n")
                    f.write(f"best_epoch,{best_epoch}\n")
                    for q in range(len(names_list)):
                        f.write(f"{names_list[q]},{best_miou_per_class[q]}\n")
                        print(f"{names_list[q]},{best_miou_per_class[q]}\n")
            else:
                print(f"setting:{config['parameter']['setting']}")
                print(f'best miou:{best_miou},epoch:{best_epoch}')

        if i == 0:pass
        elif i % config["parameter"]["ckpt_intervall"] == 0:
            # 每间隔一定迭代次数保存模型
            torch.save({"state_dict": net_student.state_dict()}, os.path.join(savedir_root, f"model_{i}.pth"), )

        if i == 0:
            ###Initial pseudo labeling
            # 初始伪标签生成
            cls_thresh_cuda = pseudo_label(net_teacher, config, target_train_loader, device, writer,
                                           names_list, i, running_conf, THRESHOLD_BETA, mapping_info)
        try:
            # 获取下一个训练数据
            _, target_data = train_iter_trg.__next__()
        except:
            # 如果迭代结束，则重新初始化迭代器并获取下一个训练数据
            train_iter_trg = enumerate(target_train_loader)
            _, target_data = train_iter_trg.__next__()

            # New epoch so, recalculate the per class threshold with new model
            # Updating to latest model
            if config["parameter"]["ema_teacher"]:
                # 如果使用 EMA 教师模型，则不更新
                pass
            else:
                # 否则，更新伪标签骨干网络
                net_teacher = make_a_deepcopy(net_student, logging)

            # Creating the per class Thresholds
            # 生成每类的阈值
            print("Creating the per class Thresholds...")
            cls_thresh_cuda = pseudo_label(net_teacher, config, target_train_loader, device, writer, names_list, i,
                                           running_conf, THRESHOLD_BETA,mapping_info)

        # 深拷贝训练数据
        new_data = copy.deepcopy(target_data)
        # 将训练数据移动到指定设备上
        target_data = dict_to_device(target_data, device)
        # 清空优化器的梯度
        optimizer.zero_grad()

        # 前向传播，获取分割输出
        _, output_seg_stu, _ = net_student.forward_mapped_learned(target_data)

        #### Entropy loss
        # 计算熵损失
        loss_ent = ent_weigth * minent_entropy_loss(output_seg_stu)
        ent_loss_thr = np.array(config["parameter"]["ent_loss_thr"]).astype(np.float64)
        ent_loss_thr = torch.from_numpy(ent_loss_thr).type(torch.FloatTensor).to(device)
        loss_ent = F.relu(loss_ent - ent_loss_thr, inplace=False)
        # 将熵损失写入 TensorBoard
        writer.add_scalar(f"training.entropy_loss", loss_ent, i)
        # 初始化分割损失
        loss_seg = loss_ent

        ###Calculation of SND factor
        # 对分割输出进行 softmax 操作并获取部分结果
        if True:
            output = F.softmax(output_seg_stu.clone().detach(), dim=1).detach()[:, :, 0]

            # With the SND criterion (soft neighborhood density)
            # 生成随机排列的索引
            output_rand = torch.randperm(output.size(0))
            # select_point = pred1_rand.shape[0]
            # 选择的点数
            select_point = 100
            # 对选择的输出进行归一化
            pred1 = F.normalize(output[output_rand[:select_point]])
            # 计算熵
            pred1_en = entropy(torch.matmul(pred1, pred1.t()) * 20)
            # 将 SND 指标写入 TensorBoard
            writer.add_scalar(f"pseudo_label.training.SND", pred1_en, i)
            # 将熵值添加到学生分数缓冲区
            stu_score_buffer.append(pred1_en.item())
            # 将数据和随机排列的索引添加到学生评估列表
            stu_eval_list.append([new_data, output_rand.cpu()])

        # 初始化阈值化的标签
        thresolded_label = None
        # 上下文管理器，禁用梯度计算
        with (torch.no_grad()):
            point_coords_full = target_data['pos'][:, 1:]
            features_full_stu = target_data['latents']
            # 前向传播，获取伪标签骨干网络的分割输出
            _, output_seg_teacher, _ = net_teacher.forward_mapped_learned_original(target_data)
            # 分离张量，避免梯度传播
            output_seg_teacher = output_seg_teacher.detach()
            initial_preds_logits_full_teacher = output_seg_teacher
            initial_preds_full_teacher = torch.argmax(initial_preds_logits_full_teacher, dim=1)
            ###DT-ST setting
            if True:
                # 对分割输出进行 softmax 操作
                output_pl_teacher = F.softmax(output_seg_teacher[:, :, 0], dim=1)
                # 更新运行置信度
                running_conf = update_running_conf(output_pl_teacher, running_conf, THRESHOLD_BETA)
                # 生成阈值化的标签
                thresolded_label, _, _ = pseudo_labels_probs(output_pl_teacher, running_conf, THRESHOLD_BETA)
                # 分离张量，避免梯度传播
                thresolded_label = thresolded_label.detach()

                # 生成阈值化标签的掩码
                mask_thresholded_label = thresolded_label != 0
                # 获取伪标签对应的真实标签
                true_labels_pseudo_labels = target_data["y"].detach()[mask_thresholded_label]
                # 生成忽略点的掩码
                mask_ignore_points_in_pseudo_labels = true_labels_pseudo_labels == 0
                # 计算伪标签中忽略点的比率
                pseudo_ignore_points_rate = np.sum(mask_ignore_points_in_pseudo_labels.detach().cpu().numpy()) / np.sum(
                    mask_thresholded_label.cpu().numpy())
                # 将伪标签中忽略点的比率写入 TensorBoard
                writer.add_scalar(f"pseudo_label.training.pseudo_ignore_points_rate", pseudo_ignore_points_rate, i)

            all_reliable_indices = []
            point_offset = 0
            all_vgi_weights = []
            # 获取每个点的VGI权重
            centers_per_scan = []
            num_experts_per_scan = []
            all_partition_values = []
            for c in range(target_data["N"].shape[0]):
                num_points_in_scan = target_data["N"][c].item()
                start_idx, end_idx = point_offset, point_offset + num_points_in_scan
                coords_scan = point_coords_full[start_idx:end_idx]

                partition_values_scan = torch.sqrt(coords_scan[:, 0] ** 2 + coords_scan[:, 1] ** 2 + coords_scan[:, 2] ** 2)

                all_partition_values.append(partition_values_scan)

                k_scan, centers_scan = sapl_module.determine_num_experts_bic(partition_values_scan)
                num_experts_per_scan.append(k_scan)
                centers_per_scan.append(centers_scan)

                initial_preds_logits_teacher = initial_preds_logits_full_teacher[start_idx:end_idx]
                initial_preds_teacher = initial_preds_full_teacher[start_idx:end_idx]

                reliable_indices_scan = sapl_module.vgi_ent_filter(
                    initial_preds_logits_teacher, initial_preds_teacher, coords_scan,
                    voxel_size=config["parameter"]["vgi_voxel_size"]
                )
                all_reliable_indices.append(reliable_indices_scan + start_idx)

                vgi_weights_scan = get_vgi_weights(
                    coords_scan, output_seg_stu[start_idx:end_idx, :, 0],
                    voxel_size=config["parameter"]["vgi_voxel_size"]
                )
                all_vgi_weights.append(vgi_weights_scan)

                point_offset = end_idx
            reliable_indices = torch.cat(all_reliable_indices)
            vgi_loss_weights = torch.cat(all_vgi_weights)
            partition_values_full = torch.cat(all_partition_values)
            # Heuristic: use anchors from the most complex scene for the whole batch
            if num_experts_per_scan:
                most_complex_idx = np.argmax(num_experts_per_scan)
                batch_anchors = centers_per_scan[most_complex_idx]
            else:  # Fallback for empty batch
                batch_anchors = torch.tensor([0.0, 1.0], device=device)

            if reliable_indices.shape[0] > 0:
                sapl_module.update_prototypes(
                    features_full_stu[reliable_indices], initial_preds_full_teacher[reliable_indices],
                    partition_values_full[reliable_indices], vgi_loss_weights[reliable_indices],
                    anchors=batch_anchors)

            # --- Step 4: Synergy Decision & Final Pseudo-Labeling ---

            logits_prob = F.softmax(initial_preds_logits_full_teacher, dim=1)
            proto_sim = F.softmax(sapl_module.get_prototype_similarities(features_full_stu, partition_values_full,
                                                    anchors=batch_anchors), dim=1)
            # Fuse opinions
            fused_confidence = (logits_prob.squeeze(-1) + proto_sim) / 2.0
            running_conf = update_running_conf(fused_confidence,running_conf, THRESHOLD_BETA)
            thresolded_label, _, _ = pseudo_labels_probs(fused_confidence, running_conf, THRESHOLD_BETA)

        # --- Weighted Loss Calculation ---
        if config["parameter"]["pl_no_mapping"]:
            # 如果不进行伪标签映射，则计算伪标签损失
            loss_pl = ce_loss_layer_all(output_seg_stu[:, :, 0], thresolded_label)
        else:
            # 否则，计算伪标签损失
            output_seg_merged = output_seg_stu[:, :, 0] @ summation_matrix
            loss_pl = ce_loss_layer(output_seg_merged, thresolded_label)

        loss_vgi_pl = loss_pl * vgi_loss_weights
        loss_vgi_pl = torch.mean(loss_vgi_pl)
        loss_vgi_pl *= pl_weigth

        writer.add_scalar(f"training.loss_vgi_pl", loss_vgi_pl, i)

        # 累加分割损失
        loss_seg = loss_seg + loss_vgi_pl

        # 将分割损失写入 TensorBoard
        writer.add_scalar(f"training.seg_loss", loss_seg, i)
        # 反向传播
        loss_seg.backward()
        # 更新模型参数
        optimizer.step()

        if config["parameter"]["lr_scheduler"]:
            # 如果使用学习率调度器，则将学习率写入 TensorBoard 并更新学习率
            writer.add_scalar(f"training.lr", optimizer.param_groups[0]["lr"], i)
            scheduler.step()

        # 删除分割损失张量，释放内存
        del loss_seg

        if config["parameter"]["fixed_update_iteration"]:
            if i % config["parameter"]["ema_update_iteration"] == 0:
                # 如果使用固定更新迭代次数，每间隔一定迭代次数更新 EMA 教师模型
                net_his_optimizer.step()
                logging.info("Updating the EMA Teacher at iteration {}".format(i))
            ## reset
            # 重置学生评估列表和学生分数缓冲区
            stu_eval_list = []
            stu_score_buffer = []

        else:
            if len(stu_score_buffer) >= 9 and int(len(stu_score_buffer) - 9) % 3 == 0:
                # 如果不使用固定更新迭代次数，满足条件时进行评估
                all_score = evel_stu(config, net_student, stu_eval_list, device, dict_with_mapping, alpha)
                # 计算评估结果的差值
                compare_res = np.array(all_score) - np.array(stu_score_buffer)
                if np.mean(compare_res > 0) > 0.5 or len(stu_score_buffer) > 30:
                    # 如果满足条件，则更新 EMA 教师模型
                    update_iter = len(stu_score_buffer)
                    net_his_optimizer.step()
                    logging.info(
                        "Updating the EMA Teacher at iteration {}, with updater iter {}".format(i, update_iter))

                    # 将更新迭代次数写入 TensorBoard
                    writer.add_scalar(f"pseudo_label.update_iteration", update_iter, i)
                    # 将学生原始分数的平均值写入 TensorBoard
                    writer.add_scalar(f"pseudo_label.stu_ori", np.array(stu_score_buffer).mean(), i)
                    # 将学生当前分数的平均值写入 TensorBoard
                    writer.add_scalar(f"pseudo_label.stu_now", np.array(all_score).mean(), i)

                    ## reset
                    # 重置学生评估列表和学生分数缓冲区
                    stu_eval_list = []
                    stu_score_buffer = []

    with open(os.path.join(savedir_root, f"best_miou.csv"), "w") as f:
        f.write(f"best_miou,{best_miou}\n")
        for q in range(len(names_list)):
            f.write(f"{names_list[q]},{best_miou_per_class[q]}\n")
            print(f"{names_list[q]},{best_miou_per_class[q]}\n")
if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='Process some integers.')
    # General settings
    # 添加通用设置的命令行参数
    parser.add_argument('--name', '-n', type=str, required=True)
    parser.add_argument('--setting', '-ds', type=str, required=True, default="NS2SK")
    parser.add_argument('--resume_path', '-p', type=str,
                        default="cvpr24_results/REP0_ns_semantic_TorchSparseMinkUNet_InterpAllRadiusNoDirsNet_1.0_trainSplit")
    parser.add_argument('--save_ckpt', '-scpt', type=bool, default=True)
    parser.add_argument('--tensorboard_folder', '-tf', type=str, default="UR")
    parser.add_argument('--bn_layer', '-l', type=str, default="standard")

    # Learning parameter
    # 添加学习参数的命令行参数
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--nb_iterations', '-i', type=int, default=20010)
    parser.add_argument('--ckpt_intervall', type=int, default=1000)
    parser.add_argument('--val_intervall', type=int, default=1000)
    parser.add_argument('--ent_weigth', '-ew', type=float, default=1.0)

    parser.add_argument('--lr_scheduler', '-ls', type=bool, default=False)
    parser.add_argument('--adaptive_weighting', '-aw', type=bool, default=False)

    # Select what to finetune
    # 添加微调设置的命令行参数
    parser.add_argument('--finetune', '-f', type=bool, default=False)
    parser.add_argument('--fintune_setting', '-fs', type=str,
                        choices=['LL', 'classic', 'll_and_scalable_finetune', 'shot_finetune', 'complete_finetune'],
                        default='LL')  #

    parser.add_argument('--prior_target', '-ps', type=bool, default=False)
    parser.add_argument('--free_bn_layer', '-b', type=bool, default=False)

    ### Pseudo-Label parameter
    # 添加伪标签参数的命令行参数
    parser.add_argument('--init_tgt_portion', type=float, default=0.20)
    parser.add_argument('--tgt_port_step', type=float, default=0.05)
    parser.add_argument('--max_tgt_port', type=float, default=0.5)

    parser.add_argument('--fixed_threshold', type=bool, default=False)
    parser.add_argument('--pl_no_mapping', type=bool,
                        default=False)  # Indicates if we should do the pseudo-labelling with the class mapping or without

    parser.add_argument('--pl_weigth', '-plw', type=float, default=1.0)
    parser.add_argument('--DEBUG_remove_ignore_points_from_pl', type=bool, default=False)

    # EMA Teacher parameter
    # 添加 EMA 教师模型参数的命令行参数
    parser.add_argument('--ema_teacher', type=bool, default=True)
    parser.add_argument('--ema_alpha', type=float, default=0.99)
    parser.add_argument('--ema_update_iteration', type=int, default=6)
    parser.add_argument('--fixed_update_iteration', type=bool, default=False)

    parser.add_argument('--ent_loss_thr', '-eth', type=float, default=0.02)
    parser.add_argument('--vgi_voxel_size', type=float, default=1.0, help="Voxel size for VGI calculation.")


    # 解析命令行参数
    opts = parser.parse_args()

    config_arguments = vars(opts)

    # 调用主函数
    main(config_arguments)
