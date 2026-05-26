import torch
import torch_scatter
import torchsparse
import torchsparse.nn.functional
from matplotlib import pyplot as plt
from torch import nn
from utils_source_free.general_imports import *  # Assuming this contains necessary imports like networks, dataloaders, etc.
from tqdm import tqdm
import copy
import argparse
import random
import numpy as np
import os


def torch_unique(x):
    unique, inverse, counts = torch.unique(x, return_inverse=True, return_counts=True)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
    inds = torch_scatter.scatter_min(perm, inverse, dim=0)[0]
    return unique, inds, inverse, counts


def get_point_in_voxel(raw_coord, voxel_size=1.0):
    voxel_grid = (raw_coord / voxel_size).int()
    hash_tensor = torch.cat((voxel_grid, torch.zeros((voxel_grid.shape[0], 1), device=voxel_grid.device)), dim=1).int()
    pc_hash = torchsparse.nn.functional.sphash(hash_tensor)
    sparse_hash, voxel_idx, inverse, voxel_point_counts = torch_unique(pc_hash)
    return sparse_hash, voxel_idx, inverse, voxel_point_counts


def get_vgi_weights(coords, preds_logits, voxel_size=1.0, beta=1.0):
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
    voxel_weights = torch.exp(-1 * beta * voxel_impurity)
    return voxel_weights[inverse]


def make_a_deepcopy(net, logging):
    net_copy = copy.deepcopy(net)
    for l_name, l_module in net_copy.named_modules():
        if isinstance(l_module, torch.nn.modules.batchnorm._BatchNorm):
            l_module.eval()
    net_copy.eval()
    return net_copy


def validation_performance_pseudo_labels(net, config, test_loader, device, cls_thresh_cuda,
                                         list_ignore_classes=[0], running_conf=None, THRESHOLD_BETA=None,
                                         mapping_information=None):
    net.eval()
    error_seg_head = 0
    pseudo_label_rate_sum = 0.0
    ignore_pseudo_label_rate_sum = 0.0
    cm_seg_head = np.zeros((config["nb_classes_inference"], config["nb_classes_inference"]))
    if mapping_information is None:
        mapping_information = sf_class_mapping_loader(source_dataset=config["source_dataset_name"],
                                                      target_dataset=config["target_dataset_name"])
    summation_matrix = summation_matrix_generator(mapping_information)
    summation_matrix = summation_matrix.to(device)

    with torch.no_grad():
        count_iter = 0
        for data in test_loader:
            data = dict_to_device(data, device)
            _, output_seg, _ = net.forward_mapped_learned(data)  # SOURCE data

            if config["parameter"]["pl_no_mapping"]:
                output_merged = output_seg[:, :, 0]
            else:
                output_merged = output_seg[:, :, 0] @ summation_matrix

            output = F.softmax(output_merged, dim=1)
            thresolded_label, _, _ = pseudo_labels_probs(output, running_conf, THRESHOLD_BETA)
            thresolded_label = thresolded_label.detach().cpu().numpy()

            output_seg_np = prediction_changer(output_seg.cpu().detach(),
                                               mapping_information)  # Only predicting on available classes
            target_seg_np = data["y"].cpu().numpy().astype(int)

            mask_pseudo_labels = thresolded_label != 0
            output_seg_np_pseudo_label = output_seg_np[mask_pseudo_labels]
            target_seg_np_pseudo_label = target_seg_np[mask_pseudo_labels]

            pseudo_label_rate_batch = 1 - (np.sum(mask_pseudo_labels) / mask_pseudo_labels.shape[0])
            pseudo_label_rate_sum += pseudo_label_rate_batch

            true_labels_pseudo_labels = data["y"].detach()[mask_pseudo_labels]
            mask_ignore_points_in_pseudo_labels = true_labels_pseudo_labels == 0
            rate_ignore_points_loss_batch = np.sum(mask_ignore_points_in_pseudo_labels.detach().cpu().numpy()) / np.sum(
                mask_pseudo_labels)
            ignore_pseudo_label_rate_sum += rate_ignore_points_loss_batch

            cm_seg_head_ = confusion_matrix(output_seg_np_pseudo_label.ravel(), target_seg_np_pseudo_label.ravel(),
                                            labels=list(range(config["nb_classes_inference"])))
            cm_seg_head += cm_seg_head_

            count_iter += 1
            if count_iter % 10 == 0:
                torch.cuda.empty_cache()

        test_seg_head_maa, accuracy_per_class = metrics.stats_accuracy_per_class(cm_seg_head,
                                                                                 ignore_list=list_ignore_classes)
        test_seg_head_miou, seg_iou_per_class = metrics.stats_iou_per_class(cm_seg_head,
                                                                            ignore_list=list_ignore_classes)
        test_seg_head_loss = error_seg_head / cm_seg_head.sum()

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
    print("Running pseudo_label()...")
    cls_thresh = np.ones(config["nb_classes"], dtype=np.float32)
    cls_thresh_cuda = torch.from_numpy(cls_thresh).to(device)

    return_data_pl = validation_performance_pseudo_labels(net_pseudo_label, config, target_train_loader, device,
                                                          cls_thresh_cuda, [0], running_conf, THRESHOLD_BETA,
                                                          mapping_information)

    logging.info(f"Pseudo Label mIoU: {return_data_pl['test_seg_head_miou']}")
    writer.add_scalar(f"pseudo_label.seg_mIou", return_data_pl['test_seg_head_miou'], i)
    logging.info(f"Pseudo Label Per class {return_data_pl['seg_iou_per_class']}")
    for q in range(len(names_list)):
        writer.add_scalar(f"pseudo_label.seg_Iou_{names_list[q]}", return_data_pl['seg_iou_per_class'][q], i)
    logging.info(f"Pseudo Label rate: {return_data_pl['pseudo_label_rate']}")
    writer.add_scalar(f"pseudo_label.pseudo_label_rate", return_data_pl['pseudo_label_rate'], i)

    logging.info(f"Ignore class rates in pseudo label {return_data_pl['ignore_points_rate']}")
    writer.add_scalar(f"pseudo_label.evaluation.pseudo_ignore_points_rate", return_data_pl['ignore_points_rate'], i)

    return cls_thresh_cuda


def pseudo_labels_probs(probs, running_conf, THRESHOLD_BETA, RUN_CONF_UPPER=0.80, ignore_augm=None, discount=True):
    RUN_CONF_UPPER = RUN_CONF_UPPER
    RUN_CONF_LOWER = 0.20
    N, C = probs.size()
    max_conf, max_idx = probs.max(1, keepdim=True)

    probs_peaks = torch.zeros_like(probs)
    probs_peaks.scatter_(1, max_idx, max_conf)

    top_peaks, _ = probs_peaks.view(N, C).max(0)
    top_peaks *= RUN_CONF_UPPER

    if discount:
        top_peaks *= (1. - torch.exp(- running_conf / THRESHOLD_BETA))

    top_peaks.clamp_(RUN_CONF_LOWER)
    probs_peaks.gt_(top_peaks.view(1, C))

    ignore = probs_peaks.sum(1, keepdim=True) != 1

    pseudo_labels = max_idx.clone()
    pseudo_labels[ignore] = 0
    pseudo_labels = pseudo_labels.squeeze(1)

    return pseudo_labels, max_conf, max_idx


def update_running_conf(probs, running_conf, THRESHOLD_BETA, tolerance=1e-8):
    STAT_MOMENTUM = 0.9
    N, C = probs.size()
    probs_avg = probs.mean(0).view(C, -1).mean(-1)
    update_index = probs_avg > tolerance
    new_index = update_index & (running_conf == THRESHOLD_BETA)
    running_conf[new_index] = probs_avg[new_index]

    running_conf *= STAT_MOMENTUM
    running_conf += (1 - STAT_MOMENTUM) * probs_avg
    return running_conf


def entropy(p, prob=True, mean=True):
    if prob:
        p = F.softmax(p, dim=1)
    en = -torch.sum(p * torch.log(p + 1e-5), 1)
    if mean:
        return torch.mean(en)
    else:
        return en


class WeightEMA(object):
    def __init__(self, params, src_params, alpha):
        self.params = list(params)
        self.src_params = list(src_params)
        self.alpha = alpha

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for p, src_p in zip(self.params, self.src_params):
            p.data.mul_(self.alpha)
            p.data.add_(src_p.data * one_minus_alpha)


def evel_stu(config, net, stu_eval_list, device, dict_with_mapping, alpha):
    net.eval()
    eval_result = []
    with torch.no_grad():
        for i, (target_data, permute_index) in enumerate(stu_eval_list):
            target_data = dict_to_device(target_data, device)
            _, output, _ = net.forward_mapped_learned(target_data)
            output = F.softmax(output, dim=1)[:, :, 0]
            pred1_rand = permute_index
            select_point = 100
            pred1 = F.normalize(output[pred1_rand[:select_point]])
            pred1_en = entropy(torch.matmul(pred1, pred1.t()) * 20)
            eval_result.append(pred1_en.item())

    net.train()
    for l_name, l_module in net.named_modules():
        if isinstance(l_module, torch.nn.modules.batchnorm._BatchNorm):
            l_module.eval()
    return eval_result


class SAPL_Module(nn.Module):
    def __init__(self, num_classes, model, alpha=0.9, gsp_sigma=10.0):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.min_feat_points = 1
        self.max_experts = 6  
        self.gsp_sigma = gsp_sigma

        initial_prototypes = copy.deepcopy(model.dual_seg_head.weight.data.squeeze(-1).T)
        self.expert_prototypes = nn.ParameterList(
            [nn.Parameter(copy.deepcopy(initial_prototypes)) for _ in range(self.max_experts)]
        )
        for p in self.expert_prototypes:
            p.data = F.normalize(p.data, p=2, dim=0)

    def _kmeans_1d_gpu(self, data, K, max_iters=10):
        if data.numel() == 0:
            return 0.0, torch.zeros(K, device=data.device)
        centers = torch.linspace(data.min(), data.max(), K, device=data.device)
        for _ in range(max_iters):
            diffs = torch.abs(data.unsqueeze(1) - centers.unsqueeze(0))
            assignments = torch.argmin(diffs, dim=1)
            new_centers = torch_scatter.scatter_mean(data, assignments, dim=0, dim_size=K)
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

    def _get_soft_partition_weights(self, partition_values, anchors):
        dist_matrix = partition_values.unsqueeze(1) - anchors.unsqueeze(0)
        weights = torch.exp(-(dist_matrix ** 2) / (2 * self.gsp_sigma ** 2))
        return weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

    def vgi_ent_filter(self, preds_logits, predictions, point_coords, voxel_size=0.5, percent=0.7):
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
    if True:
        torch.manual_seed(1234)
        random.seed(1234)
        np.random.seed(1234)

        torch.autograd.set_detect_anomaly(True)
        writer = SummaryWriter(
            'runs_sapl/{}'.format(f"{config_arguments['tensorboard_folder']}/{config_arguments['name']}"))

        if os.path.isfile(config_arguments["resume_path"]):
            file_path_config = os.path.join(os.path.dirname(config_arguments["resume_path"]), "config.yaml")
        else:
            file_path_config = os.path.join(config_arguments["resume_path"], "config.yaml")

        config = read_yaml_file(file_path_config)
        if config:
            print(file_path_config)
            print(f"Loaded Config: {config}")
        else:
            print("Failed to read the config YAML file.")

        logging.getLogger().setLevel(config["logging"])
        config["parameter"] = config_arguments
        config = config_adapter(config)
        config["ignore_class"] = 0

        mapping_info = sf_class_mapping_loader(source_dataset=config["source_dataset_name"],
                                               target_dataset=config["target_dataset_name"])
        summation_matrix = summation_matrix_generator(mapping_info)

        for k, v in config_arguments.items():
            logging.info(f"{k}: {v}")

        logging.info("Creating the network")
        logging.info(f"Self-Supervised Setting")
        config['training_batch_size'] = config["parameter"]["batch_size"]
        config["test_batch_size"] = 16
        savedir_root = f"ckpts_bn/{config_arguments['name']}"
        os.makedirs(savedir_root, exist_ok=True)
        config["ns_dataset_version"] = 'v1.0-trainval'
        config["network_backbone"] = 'TorchSparseMinkUNet_learned'

        name_shift_inverse = {}
        for key, value in name_shift.items():
            name_shift_inverse[value] = key

        config["da_fixed_head_path_model"] = config["parameter"]["resume_path"]

        device = torch.device(config['device'])
        if config["device"] == "cuda":
            torch.backends.cudnn.benchmark = True

        bb_dir_root = get_bbdir_root(config)
        latent_size = config["network_latent_size"]
        backbone = config["network_backbone"]
        decoder = {'name': config["network_decoder"], 'k': config['network_decoder_k']}

        in_channels_source, _, in_channels_target, _ = da_get_inputs(config)
        logging.info("In channels source {}".format(in_channels_source))
        logging.info("in channels target {}".format(in_channels_target))
        logging.info("Creating the network")

        def network_function():
            return networks.Network(
                in_channels=in_channels_source,
                latent_size=latent_size,
                backbone=backbone,
                voxel_size=config["voxel_size"],
                dual_seg_head=config["dual_seg_head"],
                target_in_channels=in_channels_target,
                config=config
            )

        net_student = network_function()
        if os.path.isfile(bb_dir_root):
            ckpt_path = bb_dir_root
            logging.info(f"Failed: Load ckpt from {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_new = {}
        for key in checkpoint["state_dict"].keys():
            if key in name_shift_inverse:
                checkpoint_new[name_shift_inverse[key]] = checkpoint["state_dict"][key]
            else:
                if "num_batches_tracked" in key or "point_transforms" in key:
                    pass
                else:
                    checkpoint_new[key] = checkpoint["state_dict"][key]

        try:
            net_student.load_state_dict(checkpoint_new)
        except Exception as e:
            print(e)
            logging.info(
                f"Loaded parameters do not match exactly net architecture, switching to load_state_dict strict=false")
            net_student.load_state_dict(checkpoint_new, strict=False)

        logging.info(f"Network -- Number of parameters {count_parameters(net_student)}")
        target_DatasetClass = get_dataset(eval("datasets." + config["target_dataset_name"]))
        val_number = 1
        print(f"config dataset source {config}")
        dataloader_dict = da_sf_get_dataloader(target_DatasetClass, config, net_student, network_function,
                                               val=val_number,
                                               train_shuffle=True, keep_orignal_data=True)

        target_train_loader = dataloader_dict["target_train_loader"]
        target_test_loader = dataloader_dict["target_test_loader"]

        os.makedirs(savedir_root, exist_ok=True)
        save_config_file(eval(str(config)), os.path.join(savedir_root, "config.yaml"))

        loss_layer = torch.nn.BCEWithLogitsLoss()
        weights_ss = torch.ones(config["nb_classes_inference"])
        list_ignore_classes = ignore_selection(config["ignore_idx"])
        for idx_ignore_class in list_ignore_classes:
            weights_ss[idx_ignore_class] = 0
        logging.info(f"Ignored classes {list_ignore_classes}")
        logging.info(f"Weights of the different classes {weights_ss}")
        weights_ss = weights_ss.to(device)
        ce_loss_layer = torch.nn.CrossEntropyLoss(weight=weights_ss, reduction="none")

        weights_ss_all = torch.ones(config["nb_classes"])
        list_ignore_classes = ignore_selection(config["ignore_idx"])
        for idx_ignore_class in list_ignore_classes:
            weights_ss_all[idx_ignore_class] = 0
        logging.info(f"Ignored classes {list_ignore_classes}")
        logging.info(f"Weights of the different classes for all classes {weights_ss_all}")
        weights_ss_all = weights_ss_all.to(device)
        ce_loss_layer_all = torch.nn.CrossEntropyLoss(weight=weights_ss_all, reduction="none")

        net_student.eval()
        net_student.to(device)
        dict_with_mapping = {}
        alpha = None
        list_parameter_to_update = []
        list_parameter_others = []

        net_student, list_parameter_to_update, list_parameter_others = \
            configure_freeze_models(net_student, config, list_parameter_to_update, list_parameter_others)

        for l_name, l_module in net_student.named_modules():
            if isinstance(l_module, torch.nn.modules.batchnorm._BatchNorm):
                l_module.eval()

        class_prior = np.zeros((1))
        class_prior, names_list = class_prior_class_names(config, logging)

        alpha = 0
        alpha_tensor = (torch.ones(1) * alpha).to(device)

        ent_weigth = np.array(config["parameter"]["ent_weigth"]).astype(np.float64)
        ent_weigth = torch.from_numpy(ent_weigth).type(torch.FloatTensor).to(device)
        pl_weigth = np.array(config["parameter"]["pl_weigth"]).astype(np.float64)
        pl_weigth = torch.from_numpy(pl_weigth).type(torch.FloatTensor).to(device)

        summation_matrix = summation_matrix.to(device)
        class_prior = torch.from_numpy(class_prior).to(device)

        THRESHOLD_BETA = 0.001
        running_conf = torch.zeros(config["nb_classes"]).cuda()
        running_conf.fill_(THRESHOLD_BETA)

        stu_eval_list = []
        stu_score_buffer = []
        res_dict = {'stu_ori': [], 'stu_now': [], 'update_iter': []}

        net_teacher = make_a_deepcopy(net_student, logging)
        if config["parameter"]["ema_teacher"]:
            net_his_optimizer = WeightEMA(
                list(net_teacher.parameters()),
                list(net_student.parameters()),
                alpha=0.99)

        net_student.to(device)
        if config["parameter"]["finetune"] and config["parameter"]["fintune_setting"] == "classic":
            logging.info("Classifier get updated with 10X higher LR than backbone.")
            optimizer = torch.optim.AdamW([
                {"params": list_parameter_to_update, "lr": config["parameter"]["learning_rate"]},
                {"params": list_parameter_others, "lr": config["parameter"]["learning_rate"] / 10.0}
            ])
        else:
            optimizer = torch.optim.AdamW([{"params": list_parameter_to_update}], config["parameter"]["learning_rate"])
        if config["parameter"]["lr_scheduler"]:
            scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer,
                                                              total_iters=config["parameter"]["nb_iterations"],
                                                              power=0.9, last_epoch=-1, verbose=False)

    k_candidates_list = [int(k) for k in config["parameter"]["k_candidates"].split(',')]
    sapl_module = SAPL_Module(
        num_classes=config["nb_classes"],
        model=net_student,
        gsp_sigma=config["parameter"]["gsp_sigma"]  
    )

    train_iter_trg = enumerate(target_train_loader)
    best_miou = 0

    for i in tqdm(range(config["parameter"]["nb_iterations"]), desc="Training"):
        if i % config["parameter"]["val_intervall"] == 0:
            logging.info(i)
            return_data_val_target_mapped = \
                validation_non_premap(net_student, config, target_test_loader, epoch=0, disable_log=False,
                                      device=device,
                                      list_ignore_classes=[0], mapping_information=mapping_info)
            logging.info(f"mIoU: {return_data_val_target_mapped['test_seg_head_miou']}")
            writer.add_scalar(f"validation.seg_mIou", return_data_val_target_mapped['test_seg_head_miou'], i)
            logging.info(f"Per class {return_data_val_target_mapped['seg_iou_per_class']}")
            for q in range(len(names_list)):
                writer.add_scalar(f"validation.seg_Iou_{names_list[q]}",
                                  return_data_val_target_mapped['seg_iou_per_class'][q], i)

            for l_name, l_module in net_student.named_modules():
                if isinstance(l_module, torch.nn.modules.batchnorm._BatchNorm):
                    l_module.eval()

        if i % config["parameter"]["ckpt_intervall"] == 0:
            torch.save({"state_dict": net_student.state_dict()}, os.path.join(savedir_root, f"model_{i}.pth"), )

        if i == 0:
            cls_thresh_cuda = pseudo_label(net_teacher, config, target_train_loader, device, writer,
                                           names_list, i, running_conf, THRESHOLD_BETA, mapping_info)
        try:
            _, target_data = train_iter_trg.__next__()
        except:
            train_iter_trg = enumerate(target_train_loader)
            _, target_data = train_iter_trg.__next__()

            if config["parameter"]["ema_teacher"]:
                pass
            else:
                net_teacher = make_a_deepcopy(net_student, logging)

            print("Creating the per class Thresholds...")
            cls_thresh_cuda = pseudo_label(net_teacher, config, target_train_loader, device, writer, names_list, i,
                                           running_conf, THRESHOLD_BETA, mapping_info)

        new_data = copy.deepcopy(target_data)
        target_data = dict_to_device(target_data, device)
        optimizer.zero_grad()

        _, output_seg_stu, _ = net_student.forward_mapped_learned(target_data)

        loss_ent = ent_weigth * minent_entropy_loss(output_seg_stu)
        ent_loss_thr = np.array(config["parameter"]["ent_loss_thr"]).astype(np.float64)
        ent_loss_thr = torch.from_numpy(ent_loss_thr).type(torch.FloatTensor).to(device)
        loss_ent = F.relu(loss_ent - ent_loss_thr, inplace=False)
        writer.add_scalar(f"training.entropy_loss", loss_ent, i)
        loss_seg = loss_ent


        output = F.softmax(output_seg_stu.clone().detach(), dim=1).detach()[:, :, 0]
        output_rand = torch.randperm(output.size(0))
        select_point = 100
        pred1 = F.normalize(output[output_rand[:select_point]])
        pred1_en = entropy(torch.matmul(pred1, pred1.t()) * 20)
        writer.add_scalar(f"pseudo_label.training.SND", pred1_en, i)
        stu_score_buffer.append(pred1_en.item())
        stu_eval_list.append([new_data, output_rand.cpu()])

        thresolded_label = None

        

        with (torch.no_grad()):
            point_coords_full = target_data['pos'][:, 1:]
            features_full_stu = target_data['latents']

            _, output_seg_teacher, _ = net_teacher.forward_mapped_learned_original(target_data)
            output_seg_teacher = output_seg_teacher.detach()
            initial_preds_logits_full_teacher = output_seg_teacher
            initial_preds_full_teacher = torch.argmax(initial_preds_logits_full_teacher, dim=1)


            output_pl_teacher = F.softmax(output_seg_teacher[:, :, 0], dim=1)
            running_conf = update_running_conf(output_pl_teacher, running_conf, THRESHOLD_BETA)
            thresolded_label, _, _ = pseudo_labels_probs(output_pl_teacher, running_conf, THRESHOLD_BETA)
            thresolded_label = thresolded_label.detach()
            mask_thresholded_label = thresolded_label != 0
            true_labels_pseudo_labels = target_data["y"].detach()[mask_thresholded_label]
            mask_ignore_points_in_pseudo_labels = true_labels_pseudo_labels == 0
            pseudo_ignore_points_rate = np.sum(mask_ignore_points_in_pseudo_labels.detach().cpu().numpy()) / np.sum(
                mask_thresholded_label.cpu().numpy())
            writer.add_scalar(f"pseudo_label.training.pseudo_ignore_points_rate", pseudo_ignore_points_rate, i)

            all_reliable_indices = []
            point_offset = 0
            all_vgi_weights = []
            centers_per_scan = []
            num_experts_per_scan = []
            all_partition_values = []

            for c in range(target_data["N"].shape[0]):
                num_points_in_scan = target_data["N"][c].item()
                start_idx, end_idx = point_offset, point_offset + num_points_in_scan
                coords_scan = point_coords_full[start_idx:end_idx]

                partition_values_scan = torch.sqrt(
                    coords_scan[:, 0] ** 2 + coords_scan[:, 1] ** 2 + coords_scan[:, 2] ** 2)
                all_partition_values.append(partition_values_scan)

                k_scan, centers_scan = sapl_module.determine_num_experts_bic(partition_values_scan,
                                                                             k_candidates=k_candidates_list)
                num_experts_per_scan.append(k_scan)
                centers_per_scan.append(centers_scan)
        
                initial_preds_logits_teacher = initial_preds_logits_full_teacher[start_idx:end_idx]
                initial_preds_teacher = initial_preds_full_teacher[start_idx:end_idx]

                reliable_indices_scan = sapl_module.vgi_ent_filter(
                    initial_preds_logits_teacher, initial_preds_teacher, coords_scan,
                    voxel_size=config["parameter"]["vgi_voxel_size"],
                    percent=config["parameter"]["filter_percent"]
                )
                all_reliable_indices.append(reliable_indices_scan + start_idx)

                vgi_weights_scan = get_vgi_weights(
                    coords_scan, output_seg_stu[start_idx:end_idx, :, 0],
                    voxel_size=config["parameter"]["vgi_voxel_size"],
                    beta=config["parameter"]["vgi_beta"]
                )
                all_vgi_weights.append(vgi_weights_scan)

                point_offset = end_idx

            reliable_indices = torch.cat(all_reliable_indices)
            vgi_loss_weights = torch.cat(all_vgi_weights)
            partition_values_full = torch.cat(all_partition_values)

            if num_experts_per_scan:
                most_complex_idx = np.argmax(num_experts_per_scan)
                batch_anchors = centers_per_scan[most_complex_idx]
            else:
                batch_anchors = torch.tensor([0.0, 1.0], device=device)

            if reliable_indices.shape[0] > 0:
                sapl_module.update_prototypes(
                    features_full_stu[reliable_indices], initial_preds_full_teacher[reliable_indices],
                    partition_values_full[reliable_indices], vgi_loss_weights[reliable_indices],
                    anchors=batch_anchors)

            
            logits_prob = F.softmax(initial_preds_logits_full_teacher, dim=1)
            proto_sim = F.softmax(sapl_module.get_prototype_similarities(features_full_stu, partition_values_full,
                                                                         anchors=batch_anchors), dim=1)
            fused_confidence = (logits_prob.squeeze(-1) + proto_sim) / 2.0
            running_conf = update_running_conf(fused_confidence, running_conf, THRESHOLD_BETA)
            thresolded_label, _, _ = pseudo_labels_probs(fused_confidence, running_conf, THRESHOLD_BETA)


        if config["parameter"]["pl_no_mapping"]:
            loss_pl = ce_loss_layer_all(output_seg_stu[:, :, 0], thresolded_label)
        else:
            output_seg_merged = output_seg_stu[:, :, 0] @ summation_matrix
            loss_pl = ce_loss_layer(output_seg_merged, thresolded_label)

        loss_vgi_pl = loss_pl * vgi_loss_weights
        loss_vgi_pl = torch.mean(loss_vgi_pl)
        loss_vgi_pl *= pl_weigth

        writer.add_scalar(f"training.loss_vgi_pl", loss_vgi_pl, i)
        loss_seg = loss_seg + loss_vgi_pl
        writer.add_scalar(f"training.seg_loss", loss_seg, i)
        loss_seg.backward()
        optimizer.step()

        if config["parameter"]["lr_scheduler"]:
            writer.add_scalar(f"training.lr", optimizer.param_groups[0]["lr"], i)
            scheduler.step()

        del loss_seg

        if config["parameter"]["fixed_update_iteration"]:
            if i % config["parameter"]["ema_update_iteration"] == 0:
                net_his_optimizer.step()
                logging.info("Updating the EMA Teacher at iteration {}".format(i))
            stu_eval_list = []
            stu_score_buffer = []

        else:
            if len(stu_score_buffer) >= 9 and int(len(stu_score_buffer) - 9) % 3 == 0:
                all_score = evel_stu(config, net_student, stu_eval_list, device, dict_with_mapping, alpha)
                compare_res = np.array(all_score) - np.array(stu_score_buffer)
                if np.mean(compare_res > 0) > 0.5 or len(stu_score_buffer) > 30:
                    update_iter = len(stu_score_buffer)
                    net_his_optimizer.step()
                    logging.info(
                        "Updating the EMA Teacher at iteration {}, with updater iter {}".format(i, update_iter))

                    writer.add_scalar(f"pseudo_label.update_iteration", update_iter, i)
                    writer.add_scalar(f"pseudo_label.stu_ori", np.array(stu_score_buffer).mean(), i)
                    writer.add_scalar(f"pseudo_label.stu_now", np.array(all_score).mean(), i)

                    stu_eval_list = []
                    stu_score_buffer = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--name', '-n', type=str, required=True)
    parser.add_argument('--setting', '-ds', type=str, required=True, default="Synth2POSS")
    parser.add_argument('--resume_path', '-p', type=str,
                        default="./source_models/synth_semantic_TorchSparseMinkUNet")
    parser.add_argument('--save_ckpt', '-scpt', type=bool, default=True)
    parser.add_argument('--tensorboard_folder', '-tf', type=str, default="Synth2POSS")
    parser.add_argument('--bn_layer', '-l', type=str, default="scaling_per_channel")

    parser.add_argument('--learning_rate', '-lr', type=float, default=0.0025)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--nb_iterations', '-i', type=int, default=20010)
    parser.add_argument('--ckpt_intervall', type=int, default=1000)
    parser.add_argument('--val_intervall', type=int, default=1000)
    parser.add_argument('--ent_weigth', '-ew', type=float, default=1.0)
    parser.add_argument('--lr_scheduler', '-ls', type=bool, default=True)
    parser.add_argument('--adaptive_weighting', '-aw', type=bool, default=False)

    parser.add_argument('--finetune', '-f', type=bool, default=True)
    parser.add_argument('--fintune_setting', '-fs', type=str,
                        choices=['LL', 'classic', 'll_and_scalable_finetune', 'shot_finetune', 'complete_finetune'],
                        default='classic')

    parser.add_argument('--prior_target', '-ps', type=bool, default=False)
    parser.add_argument('--free_bn_layer', '-b', type=bool, default=False)

    parser.add_argument('--init_tgt_portion', type=float, default=0.20)
    parser.add_argument('--tgt_port_step', type=float, default=0.05)
    parser.add_argument('--max_tgt_port', type=float, default=0.5)

    parser.add_argument('--fixed_threshold', type=bool, default=False)
    parser.add_argument('--pl_no_mapping', type=bool, default=True)
    parser.add_argument('--pl_weigth', '-plw', type=float, default=1.0)
    parser.add_argument('--DEBUG_remove_ignore_points_from_pl', type=bool, default=False)

    parser.add_argument('--ema_teacher', type=bool, default=True)
    parser.add_argument('--ema_alpha', type=float, default=0.99)
    parser.add_argument('--ema_update_iteration', type=int, default=6)
    parser.add_argument('--fixed_update_iteration', type=bool, default=False)
    parser.add_argument('--ent_loss_thr', '-eth', type=float, default=0.02)
    parser.add_argument('--vgi_voxel_size', type=float, default=1.0, help="Voxel size for VGI calculation.")

 
    parser.add_argument('--gsp_sigma', type=float, default=10.0, help="Gaussian kernel bandwidth (sigma) in AEI")
    parser.add_argument('--vgi_beta', type=float, default=1.0, help="VGI scaling factor (beta) in LSC")
    parser.add_argument('--filter_percent', type=float, default=0.7, help="Reliability filtering ratio threshold")
    parser.add_argument('--k_candidates', type=str, default="2,3,4,5,6",
                        help="Candidate set K, comma separated (e.g. 2,3,4,5,6)")


    opts = parser.parse_args()
    config_arguments = vars(opts)
    main(config_arguments)
