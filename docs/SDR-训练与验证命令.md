# SDR-Voxel R-CNN 训练与验证命令手册

> 所有命令都在 **项目根目录** `KD_Voxel/` 下执行：
> ```bash
> cd C:/Users/86184/Desktop/稀疏点云检测/KD_Voxel
> ```
> 当前数据：`train 400 帧 / val 100 帧`，**infos 与 gt_database 尚未生成**，必须先做第 0 节。

---

## 0. 一次性准备（只做一次，约 10~30 分钟）

### 0.1 编译 CUDA 算子

```bash
python setup.py develop
```

**干什么**：编译 `roiaware_pool3d`、`iou3d` 等 CUDA 扩展。训练和评测里的「点是否在框内」
「3D IoU」都靠它。没编译时连 `import pcdet` 都会失败。

---

### 0.2 生成 KITTI infos 与 GT 数据库

```bash
python tools/gen_kitti_infos_local.py
```

**干什么**：扫描 `data/kitti/training/` 的标定、标注、点云，产出 4 个文件：

| 文件 | 内容 | 谁在用 |
|---|---|---|
| `data/kitti/kitti_infos_train.pkl` | 400 帧的路径、标定、GT 框、点数 | 训练集 |
| `data/kitti/kitti_infos_val.pkl` | 100 帧同上 | 验证/评测 |
| `data/kitti/kitti_infos_trainval.pkl` / `_test.pkl` | 合并集 / 测试集 | 一般不用 |
| `data/kitti/gt_database/` + `kitti_dbinfos_train.pkl` | 抠出来的单个 GT 点云 | `gt_sampling` 数据增强 |

**为什么用这个脚本而不是官方命令**：官方命令要先编译好 CUDA 才能 `import pcdet.datasets`，
而这一步只需要 CPU 版的 `points_in_boxes_cpu`。本脚本用数值等价的 NumPy 实现替换它，
所以在 `setup.py develop` 之前或之后都能跑。

> ⚠️ **只需生成一次，六组实验共用**。`kitti_dataset.yaml` / `_sparse.yaml` / `_sdr.yaml`
> 三个配置的 `INFO_PATH` 都指向同一份 `kitti_infos_train.pkl`，不要为不同预设重复生成。

---

### 0.3 三项开跑前自检（都不需要 GPU）

```bash
# (a) 文档 ↔ 代码一致性：64 条断言
python tools/check_sdr_conformance.py

# (b) 教师 ckpt 体检：张量是否齐全、是不是改进模型冒充的
python tools/check_teacher_ckpt.py \
       --ckpt checkpoints/voxel_rcnn_84.54.pth \
       --config tools/cfgs/kitti_models/sdr_voxel_rcnn_car.yaml

# (c) 五步链路自检：配置 → 数据 → 模型 → 前向 → 蒸馏损失
python tools/check_sdr_setup.py --preset sdr
```

**期望输出**：

- (a) `合计 64 项：PASS 63 / FAIL 0 / WARN 1`（WARN 是 Car-only 单类训练，代码没错）
- (b) `[PASS] 教师侧张量 156 个全部命中`
- (c) 最后一步打印六个监控量的初值 + 门控熵 + 各尺度权重 + EIP-BS 保留率

(c) 是最有价值的一条：它真的建了模型、真的跑了一次前向和反向，
能提前抓出通道不匹配、dtype 不一致这类只在运行时才暴露的问题。

---

## 1. 冒烟测试（先花 10 分钟确认能跑通）

```bash
python tools/train.py --preset sdr --epochs 1 --batch_size 2 \
       --extra_tag smoke --no_plot_after_train --no_vis_after_train
```

**干什么**：只跑 1 个 epoch（400 帧 / batch 2 = 200 iter），确认四件事：

1. 启动日志出现 `teacher 权重已载入: checkpoints/voxel_rcnn_84.54.pth (156/156 个张量匹配, 100%)`
2. 训练中每 50 iter 出现两行 `[SDR]`（roi_ratio / gate_eff / theta_mean / kd_w / gate_entropy …）
3. loss 在下降，tqdm 在前进
4. 显存没爆

三个参数为什么这么给：

- `--epochs 1`：只是验证管线，不要看 AP
- `--no_plot_after_train --no_vis_after_train`：跳过出图，省时间；正式训练时**不要加**
- `--extra_tag smoke`：输出到独立目录，不污染正式实验

---

## 2. 主实验（三组对比，论文主表）

```bash
# ① 基线：稠密 64 线点云，原始 Voxel R-CNN
python tools/train.py --preset baseline --batch_size 2 --epochs 80 --extra_tag r1

# ② 稀疏对照：同一个基线模型，但输入被均匀抽线到 25%
python tools/train.py --preset sparse   --batch_size 2 --epochs 80 --extra_tag r1

# ③ 改进模型：完整 SDR-Voxel R-CNN（EIP-BS + MS-DGE + CG-RD）
python tools/train.py --preset sdr      --batch_size 2 --epochs 80 --extra_tag r1 --eval_interval 10
```

**三组的关系**（消融只有一个自变量）：

| 组 | 输入 | 模型 | 说明 |
|---|---|---|---|
| `baseline` | 稠密 64 线 | 原版 Voxel R-CNN | 上限参照 |
| `sparse` | EIP-BS 后 25% | **原版**（不改一行代码） | 只换输入，看掉多少 |
| `sdr` | EIP-BS 后 25% | MSDGE + CG-RD | 看能不能补回来 |

`sparse` 用的是「未经改动的 Voxel R-CNN」，靠 `WRITE_TO_MAIN: true` 把稀疏结果写成主键。
所以 `sparse → sdr` 的差值，全部来自两个创新点，不掺杂任何结构差异。

`--eval_interval 10` 每 10 个 epoch 跑一次验证并写 `val_metrics.csv`，
画 AP-epoch 曲线用；不需要就删掉（会快一些）。

---

## 3. 消融实验（回答「每个创新点各自值多少」）

```bash
# ② 换成归一化 L2 距离（验证「黎曼」比「普通 L2」好）
python tools/train.py --preset sdr_mse  --batch_size 2 --epochs 80 --extra_tag r1

# 关掉蒸馏（①+③ 的净贡献）
python tools/train.py --preset sdr_nokd --batch_size 2 --epochs 80 --extra_tag r1

# ① 换回 MeanVFE（②+③ 的净贡献）
python tools/train.py --preset sdr_noms --batch_size 2 --epochs 80 --extra_tag r1
```

| 预设 | 与 `sdr` 的唯一差别 | 能回答什么 |
|---|---|---|
| `sdr_mse` | `DIST_TYPE: riemann → l2n` | 黎曼几何本身值多少 |
| `sdr_nokd` | `KD.ENABLE: False` | 蒸馏这一路值多少 |
| `sdr_noms` | `VFE: MSDGE → MeanVFE` | 密度门控编码值多少 |

**一条命令跑完六组**（耗时长，建议挂后台）：

```bash
for p in baseline sparse sdr sdr_mse sdr_nokd sdr_noms; do
  python tools/train.py --preset $p --batch_size 2 --epochs 80 --extra_tag r1
done
```

---

## 4. 单独评测

训练脚本结束时会**自动**评测最后一个 ckpt，一般不需要手跑。以下用于换 ckpt、换指标、或补评：

```bash
# 评测某个 ckpt，并额外算 AP@0.5:0.95 与 3D P/R/F1
python tools/test.py --preset sdr --batch_size 2 \
       --ckpt output/cfgs/kitti_models/sdr_voxel_rcnn_car/r1/ckpt/checkpoint_epoch_80.pth \
       --extra_metrics ap50_95 prf

# 评测目录下所有 epoch 的 ckpt（画 AP-epoch 曲线用）
python tools/test.py --preset sdr --batch_size 2 --eval_all \
       --extra_tag r1

# 测推理时延（单帧毫秒数，写进论文效率表）
python tools/test.py --preset sdr --batch_size 1 --infer_time \
       --ckpt output/cfgs/kitti_models/sdr_voxel_rcnn_car/r1/ckpt/checkpoint_epoch_80.pth
```

> ⚠️ **`--preset` 必须与训练时一致**。`sdr` 的 VFE 是 64 通道 MSDGE，`baseline` 是 4 通道
> MeanVFE，ckpt 完全不通用。用错 preset 会在加载权重时形状不匹配报错。

---

## 5. 训练结束的产出（自动生成，不需额外命令）

以 `--preset sdr --extra_tag r1` 为例，全部在：

```
output/cfgs/kitti_models/sdr_voxel_rcnn_car/r1/
├── train_YYYYMMDD-HHMMSS.log        终端日志（含 [SDR] 行）
├── iter_log.csv                     每 iter 的全部 tb_dict（画图数据源）
├── val_metrics.csv                  --eval_interval 的 AP 记录
├── training_curves.png              loss / mAP 曲线
├── sdr_diagnostics.png              3×3 九格诊断面板
├── sdr_summary.json                 首值/末值/均值/极值
├── ckpt/checkpoint_epoch_*.pth      权重
├── eval/eval_with_train/            评测结果 + PR 曲线
├── tensorboard/                     tensorboard 日志
└── visualizations/
    ├── det_<frame_id>.png           每帧三联图：GT / Pred / 放大叠加
    ├── detections_overview.png      2×K 总览
    └── detections_summary.json      帧号、点数、保留率、预测数、最高分
```

### 5.1 事后补图（训练被中断、或想换平滑窗口）

```bash
# 只重画统计图
python tools/plot_curves.py \
       --exp_dir output/cfgs/kitti_models/sdr_voxel_rcnn_car/r1 \
       --skip_head 500 --smooth 20

# 只补检测效果图
python tools/visualize_detections.py --preset sdr \
       --ckpt output/cfgs/kitti_models/sdr_voxel_rcnn_car/r1/ckpt/checkpoint_epoch_80.pth \
       --num_samples 3
```

`--skip_head 500` 跳过前 500 iter：蒸馏权重有 ramp-up，头部会把均值拉偏。

### 5.2 检测效果图的帧号是固定的

首次出图时随机抽 `--vis_num` 帧，写进 `output/vis_frames.json`（全仓库共享）；
之后 baseline / sparse / sdr / sdr_mse **都读这份清单**，出的图是同一批场景，可以并排贴进论文。

```bash
# 换一批帧
python tools/train.py --preset sdr --reset_frames --vis_num 3 ...

# 或直接指定
python tools/train.py --preset sdr --vis_frames 192 238 318 ...
```

---

## 6. 输出目录对照表

`<cfg_stem>[_<suffix>]/<extra_tag>`，所以六组天然不冲突，**同一组跑多次才需要换 `--extra_tag`**：

| 命令 | 输出目录 |
|---|---|
| `--preset baseline` | `output/cfgs/kitti_models/voxel_rcnn_car/<tag>` |
| `--preset sparse` | `output/cfgs/kitti_models/voxel_rcnn_car_sparse/<tag>` |
| `--preset sdr` | `output/cfgs/kitti_models/sdr_voxel_rcnn_car/<tag>` |
| `--preset sdr_mse` | `output/cfgs/kitti_models/sdr_voxel_rcnn_car_sdr_mse/<tag>` |
| `--preset sdr_nokd` | `output/cfgs/kitti_models/sdr_voxel_rcnn_car_sdr_nokd/<tag>` |
| `--preset sdr_noms` | `output/cfgs/kitti_models/sdr_voxel_rcnn_car_sdr_noms/<tag>` |

---

## 7. 常用参数速查

| 参数 | 默认 | 说明 |
|---|---|---|
| `--preset` | — | 六选一，一条命令切实验组；不能与 `--cfg_file` 同用 |
| `--batch_size` | 2（来自 yaml） | 单卡 batch。双分支吃显存，24G 卡可试 4 |
| `--epochs` | 80 | 冒烟用 1 |
| `--extra_tag` | `default` | 实验标签，决定子目录名 |
| `--ckpt` | — | 断点续训：指向 `checkpoint_epoch_N.pth` |
| `--eval_interval` | 0 | 每 N epoch 验证一次，0 = 只在最后验证 |
| `--num_epochs_to_eval` | 0 | 只评测最后 N 个 epoch 的 ckpt |
| `--fix_random_seed` | 关 | 固定随机种子（复现实验用） |
| `--workers` | 4 | dataloader 进程数，Windows 上若报共享内存错误就设 0 |
| `--logger_iter_interval` | 50 | `[SDR]` 日志行与 csv 的写入间隔 |
| `--no_plot_after_train` | 关 | 跳过训练后统计图 |
| `--no_vis_after_train` | 关 | 跳过训练后检测效果图 |
| `--vis_num` / `--vis_seed` | 3 / 0 | 效果图帧数 / 首次抽样种子 |
| `--reset_frames` | 关 | 丢掉固定帧清单，重新随机 |
| `--use_amp` | 关 | 混合精度。spconv 部分会自动保持 fp32 |

---

## 8. 容易踩的坑

1. **没生成 infos 就跑训练** → `FileNotFoundError: kitti_infos_train.pkl`。
   先做 0.2。
2. **为 sparse / sdr 单独生成一份 infos** → 不必要，也容易让三组数据不一致。
   三个配置共用同一份。
3. **评测时 preset 与训练不一致** → 权重形状不匹配。`sdr` 训的就只能用 `--preset sdr` 评。
4. **同一 preset 跑两次都用 `--extra_tag default`** → 第二次的 ckpt 覆盖第一次。
   换 `--extra_tag r2`。
5. **忘了教师 ckpt** → 已在 `sdr_voxel_rcnn_car.yaml` 里写死为
   `checkpoints/voxel_rcnn_84.54.pth`，不需要命令行传参；换教师只改 yaml 那一行。
   若路径失效，构建时会打印三段式警告（原因 + 后果 + 修复顺序），
   曲线照样下降但 AP 只等于稀疏基线 —— 这种失败最容易误判成「蒸馏没用」。
6. **拿 500 帧子集的 AP 和 84.54 比** → 不能比。84.54 是完整 KITTI（3712 帧）训出来的，
   你这里只有 400 帧训练、80 epoch ≈ 16000 iter，绝对 AP 一定低。
   消融结论只看**同一子集内部**的各组相对高低。
7. **Windows 上 dataloader 报错** → `--workers 0` 单进程跑，慢但不会出共享内存问题。
8. **`[SDR]` 行里 `fg_keep=-1.000`** → 该帧没有 GT（正常的哨兵值，不是 bug），
   统计时会被过滤，不参与均值。
