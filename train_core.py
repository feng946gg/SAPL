from utils_source_free.general_imports import *


""""
setting command (NS2SK, Synth2SK, Synth2POSS, NS2POSS, NS2PD, NS2WY)

python train_ttyd_core.py --name="TTYD_core_synth_sk" --bn_layer="scaling_per_channel" 
 --resume_path=source_models/synth_semantic_TorchSparseMinkUNet --setting='Synth2SK'
 --learning_rate=0.00001 --ent_loss_thr=0.02 --div_loss_thr=0.02 --tensorboard_folder='TTYD_core'
"""
def main(config_arguments):
    """
    主函数，用于设置训练环境、加载模型、定义损失函数和优化器，并执行训练和验证过程。

    :param config_arguments: 包含配置参数的字典，从命令行解析得到
    """
    #Setting the seeds
    torch.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)

    # 启用自动求导异常检测，方便调试
    torch.autograd.set_detect_anomaly(True)
    # 初始化 TensorBoard 写入器，用于记录训练和验证过程中的指标
    writer = SummaryWriter(
        'runs_eccv/{}'.format(f"{config_arguments['tensorboard_folder']}/{config_arguments['name']}"))

    # 构建配置文件的路径
    file_path_config = os.path.join(config_arguments["resume_path"], "config.yaml")
    # 读取配置文件
    config = read_yaml_file(file_path_config)

    # 检查配置文件是否成功读取
    if config is not None:
        # 打印配置文件路径和加载的配置信息
        print(file_path_config)
        print(f"Loaded Config: {config}")
    else:
        # 若读取失败，打印错误信息并返回 0
        print("Failed to read the config YAML file.")
        return 0

    # 设置日志级别
    logging.getLogger().setLevel(config["logging"])
    # 将命令行参数添加到配置字典中
    config["parameter"] = config_arguments

    ##############################################################################################

    # 选择使用的设置
    # 调整配置以适应不同的设置
    config = config_adapter(config)
    # 设置要忽略的类别
    config["ignore_class"] = 0
    # 加载源类到目标类的映射信息
    mapping_info = sf_class_mapping_loader(source_dataset=config["source_dataset_name"],
                                           target_dataset=config["target_dataset_name"])
    # 生成求和矩阵，用于类别映射
    summation_matrix = summation_matrix_generator(mapping_info)
    ##############################################################################################

    # 遍历命令行参数并记录日志
    for k,v in config_arguments.items():
        logging.info(f"{k}: {v}")

    # 记录创建网络的信息
    logging.info("Creating the network")

    # 设置训练批次大小
    config['training_batch_size'] = config["parameter"]["batch_size"]
    # 设置测试批次大小
    config["test_batch_size"] = 16
    # 定义保存模型的根目录
    savedir_root = f"ckpts_bn/{config_arguments['name']}"
    # 创建保存模型的根目录，如果已存在则不报错
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
        # 如果使用 CUDA 设备，启用 cuDNN 基准模式以提高性能
        torch.backends.cudnn.benchmark = True

    # 获取骨干网络的根目录
    bb_dir_root = get_bbdir_root(config)

    # create the network
    latent_size = config["network_latent_size"]
    backbone = config["network_backbone"]

    in_channels_source, _, in_channels_target, _ = da_get_inputs(config)
    logging.info("Creating the network")

    def network_function():
        """
        定义创建网络的函数。

        :return: 实例化的网络模型
        """
        return networks.Network(in_channels=in_channels_source,
                                latent_size=latent_size,
                                backbone=backbone,
                                voxel_size=config["voxel_size"],
                                dual_seg_head = config["dual_seg_head"],
                                target_in_channels=in_channels_target,
                                config=config)

        # 创建最终的网络模型

    net_final = network_function()

    # 构建检查点文件的路径
    ckpt_path = os.path.join(bb_dir_root, 'source_only.pth')
    # 记录加载检查点的信息
    logging.info(f"CKPT -- Load ckpt from {ckpt_path}")

    # 加载骨干网络的检查点
    checkpoint = torch.load(ckpt_path, map_location=device)

    # 更新检查点字典
    checkpoint_new = {}
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
        # 尝试严格加载更新后的检查点
        net_final.load_state_dict(checkpoint_new)
    except Exception as e:
        # 若加载失败，记录信息并尝试非严格加载
        logging.info(
            f"Loaded parameters do not match exactly net architecture, switching to load_state_dict strict=false.")
        net_final.load_state_dict(checkpoint_new, strict=False)

    # 记录网络的参数数量
    logging.info(f"Network -- Number of parameters {count_parameters(net_final)}")

    # 获取目标数据集类
    target_DatasetClass = get_dataset(eval("datasets." + config["target_dataset_name"]))

    # 验证集编号，1: 验证分割，2: 训练分割，其他: 测试分割
    val_number = 1

    # 获取数据加载器字典
    dataloader_dict = da_sf_get_dataloader(target_DatasetClass, config, net_final, network_function,
                                           val=val_number, train_shuffle=True, keep_orignal_data=False)

    # 获取目标训练数据加载器
    target_train_loader = dataloader_dict["target_train_loader"]
    # 获取目标测试数据加载器
    target_test_loader = dataloader_dict["target_test_loader"]

    # 创建保存模型的根目录，如果已存在则不报错
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
    ce_loss_layer = torch.nn.CrossEntropyLoss(weight=weights_ss)

    # 将网络设置为评估模式
    net_final.eval()
    # 将网络移动到指定设备上
    net_final.to(device)

    # 初始化要更新的参数列表
    list_parameter_to_update = []
    # 初始化其他参数列表，例如用于不同部分的参数缩放
    list_parameter_others = []

    # 配置冻结模型，并获取要更新的参数列表
    net_final, list_parameter_to_update, list_parameter_others = \
        configure_freeze_models(net_final, config, list_parameter_to_update, list_parameter_others)

    # 遍历网络的所有模块
    for l_name, l_module in net_final.named_modules():
        if isinstance(l_module, torch.nn.modules.batchnorm._BatchNorm):
            # 如果模块是批量归一化层，则将其设置为评估模式
            l_module.eval()

    # 初始化类先验
    class_prior = np.zeros((1))
    # 获取类先验和类别名称列表
    class_prior, names_list = class_prior_class_names(config, logging)

    # 重新归一化类先验，使其总和精确为 1
    class_prior = class_prior / np.sum(class_prior)
    # 记录使用的类分布
    logging.info(f"We use a distribution of {class_prior}")

    # 将熵损失阈值转换为张量并移动到指定设备上
    ent_loss_thr = np.array(config["parameter"]["ent_loss_thr"]).astype(np.float64)
    ent_loss_thr = torch.from_numpy(ent_loss_thr).type(torch.FloatTensor).to(device)

    # 将多样性损失阈值转换为张量并移动到指定设备上
    div_loss_thr = np.array(config["parameter"]["div_loss_thr"]).astype(np.float64)
    div_loss_thr = torch.from_numpy(div_loss_thr).type(torch.FloatTensor).to(device)

    # 将求和矩阵移动到指定设备上
    summation_matrix = summation_matrix.to(device)

    # 将类先验转换为张量并移动到指定设备上
    class_prior = torch.from_numpy(class_prior).to(device)

    # 创建 KL 散度损失层
    kl_loss = torch.nn.KLDivLoss(reduction="batchmean")

    # 将网络移动到指定设备上
    net_final.to(device)

    if config["parameter"]["finetune"] and config["parameter"]["fintune_setting"] == "classic":
        # 如果进行微调且设置为经典模式，分类器使用 10 倍于骨干网络的学习率
        logging.info("Classifier get updated with 10X higher LR than backbone.")
        optimizer = torch.optim.AdamW([
            {"params": list_parameter_to_update, "lr": config["parameter"]["learning_rate"]},
            {"params": list_parameter_others, "lr": config["parameter"]["learning_rate"] / 10.0}
        ])
    else:
        # 否则，使用统一的学习率
        optimizer = torch.optim.AdamW([{"params": list_parameter_to_update}], config["parameter"]["learning_rate"])

    if config["parameter"]["lr_scheduler"]:
        # 如果使用学习率调度器，初始化余弦退火学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20005, eta_min=0)

    # 记录最终优化的网络参数数量
    logging.info(f"Network -- Number of finally optimized parameters {count_parameters(net_final)}")
    # 获取目标训练数据加载器的迭代器
    train_iter_trg = enumerate(target_train_loader)

    # 开始训练迭代
    for i in range(config["parameter"]["nb_iterations"]):
        if i % config["parameter"]["val_intervall"] == 0:
            # 每间隔一定迭代次数进行验证
            logging.info(i)

            # 进行非预映射的验证
            return_data_val_target_mapped = \
                validation_non_premap(net_final, config, target_test_loader, epoch=0, disable_log=False, device=device,
                                      list_ignore_classes=[0])
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
                # 验证后，将批量归一化层设置为评估模式
            for l_name, l_module in net_final.named_modules():
                if isinstance(l_module, torch.nn.modules.batchnorm._BatchNorm):
                    l_module.eval()

        if i % config["parameter"]["ckpt_intervall"] == 0:
            # 每间隔一定迭代次数保存模型
            torch.save({"state_dict": net_final.state_dict()}, os.path.join(savedir_root, f"model_{i}.pth"))

        try:
            # 获取下一个训练数据
            _, target_data = train_iter_trg.__next__()
        except:
            # 如果迭代结束，重新初始化迭代器并获取下一个训练数据
            train_iter_trg = enumerate(target_train_loader)
            _, target_data = train_iter_trg.__next__()

        # 将训练数据移动到指定设备上
        target_data = dict_to_device(target_data, device)
        # 清空优化器的梯度
        optimizer.zero_grad()
        # 前向传播，获取分割输出
        _, output_seg, _ = net_final.forward_mapped_learned(target_data)

        # ------------------------------loss----------------------------------------------
        # 初始化分割损失
        loss_seg = None
        # 计算熵损失
        loss_ent = minent_entropy_loss(output_seg)
        # 对熵损失进行裁剪
        loss_ent = F.relu(loss_ent - ent_loss_thr, inplace=False)
        loss_seg = loss_ent
        # 将熵损失写入 TensorBoard
        writer.add_scalar(f"training.entropy_loss", loss_seg, i)

        # 计算多样性损失
        nb_points = output_seg.shape[0]
        # 映射到新的类别输出
        output_seg = output_seg[:, :, 0] @ summation_matrix
        # 计算输入的 softmax 并求平均
        input = F.softmax(output_seg[:, 1:], dim=1).sum(dim=0) / nb_points
        # 计算输入的对数
        input_log = torch.log(input)
        # 计算 KL 散度损失
        loss_kl = kl_loss(input_log, class_prior).type(torch.FloatTensor)
        # 对 KL 散度损失进行裁剪
        loss_kl = F.relu(loss_kl - div_loss_thr, inplace=False)
        div_loss = loss_kl
        # 将多样性损失写入 TensorBoard
        writer.add_scalar(f"training.diversity_loss", div_loss, i)
        # 累加分割损失
        loss_seg = loss_seg + div_loss

        # ------------------------------loss----------------------------------------------

        # 将分割损失写入 TensorBoard
        writer.add_scalar(f"training.seg_loss", loss_seg, i)
        # 反向传播
        loss_seg.backward()
        # 更新模型参数
        optimizer.step()

        # 删除分割损失张量，释放内存
        del loss_seg


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    #General settings
    parser.add_argument('--name', '-n', type=str, required=True)
    parser.add_argument('--setting', '-ds', type=str, required=True, default="NS2SK")
    parser.add_argument('--resume_path', '-p', type=str, default="cvpr24_results/REP0_ns_semantic_TorchSparseMinkUNet_InterpAllRadiusNoDirsNet_1.0_trainSplit")
    parser.add_argument('--tensorboard_folder', '-tf', type=str, default="DASF")
    parser.add_argument('--bn_layer', '-l', type=str, default="standard")

    #Learning parameter
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001)
    parser.add_argument('--batch_size', '-bs', type=int, default=4)
    parser.add_argument('--nb_iterations', '-i', type=int, default=20010)
    parser.add_argument('--ckpt_intervall', type=int, default=1000)
    parser.add_argument('--val_intervall', type=int, default=1000)
    parser.add_argument('--lr_scheduler', '-ls', type=bool, default=False)


    #Select what to finetune
    parser.add_argument('--finetune', '-f', type=bool, default=False)
    parser.add_argument('--fintune_setting', '-fs', type=str, choices=['LL', 'classic', 'll_and_scalable_finetune', 'shot_finetune', 'complete_finetune'],  default='LL') #

    #Clipping the loss
    parser.add_argument('--ent_loss_thr', '-eth', type=float, default=0.04)
    parser.add_argument('--div_loss_thr', '-dth', type=float, default=0.04)



    opts = parser.parse_args()

    config_arguments = {}
    #Experiment credentials
    config_arguments["name"] = opts.name
    config_arguments["tensorboard_folder"] = opts.tensorboard_folder
    config_arguments["resume_path"] = opts.resume_path

    #Training settings
    config_arguments["setting"] = opts.setting
    config_arguments["bn_layer"] = opts.bn_layer
    config_arguments["finetune"] = opts.finetune
    config_arguments["fintune_setting"] = opts.fintune_setting


    config_arguments["learning_rate"] = opts.learning_rate
    config_arguments["batch_size"]= opts.batch_size
    config_arguments["nb_iterations"] = opts.nb_iterations

    #Evaluation settings
    config_arguments["ckpt_intervall"] = opts.ckpt_intervall
    config_arguments["val_intervall"] = opts.val_intervall

    #Clipping
    config_arguments["ent_loss_thr"] = opts.ent_loss_thr
    config_arguments["div_loss_thr"] = opts.div_loss_thr
    config_arguments["lr_scheduler"] = opts.lr_scheduler


    main(config_arguments)