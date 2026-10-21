"""Additional evaluation metrics for KITTI-style 3D detection.

All evaluation here runs on the *KITTI* dataset only -- there is no COCO
dataset involved anywhere in this project. The two metrics below reuse the
official KITTI evaluation code, so they keep the same gt cleaning / ignore /
difficulty protocol as the reported KITTI AP and can be compared side by side:

1. ``compute_ap_50_95`` (alias ``compute_coco_map``) -- AP averaged over an IoU
   sweep (0.5:0.95 for Car) instead of the single KITTI IoU 0.7. "COCO-style"
   only refers to *how the IoU thresholds are swept*, exactly like
   ``do_coco_style_eval`` in the official KITTI devkit; the data is still KITTI.
   Reported as ``ap50_95/Car_3d/{easy,moderate,hard}``.

2. ``compute_prf_3d`` -- 3D Precision / Recall / F1 at a fixed IoU and score
   threshold. Official AP integrates a PR curve and hides the operating point;
   this gives the single working point, so you can tell whether a model is
   penalised by false positives or by misses.

Neither function touches the CUDA ops (no iou3d_nms / roiaware_pool3d), only
numpy + numba from ``kitti_object_eval_python``.
"""

import numpy as np

# index order must match kitti_object_eval_python/eval.py
NAME_TO_CLASS = {
    'Car': 0,
    'Pedestrian': 1,
    'Cyclist': 2,
    'Van': 3,
    'Person_sitting': 4,
    'Truck': 5,
}
ID_TO_NAME = {v: k for k, v in NAME_TO_CLASS.items()}

# same as get_coco_eval_result(): [min_iou, max_iou, min_height]
CLASS_TO_RANGE = {
    0: [0.5, 0.95, 10],
    1: [0.25, 0.7, 10],
    2: [0.25, 0.7, 10],
    3: [0.5, 0.95, 10],
    4: [0.25, 0.7, 10],
    5: [0.5, 0.95, 10],
}

DIFFICULTY_NAMES = ['easy', 'moderate', 'hard']


def _class_ids(class_names):
    ids = []
    for name in class_names:
        if name in NAME_TO_CLASS:
            ids.append(NAME_TO_CLASS[name])
        else:
            # skip classes the KITTI eval does not know about (e.g. 'Tram')
            continue
    return ids


def compute_ap_50_95(gt_annos, dt_annos, class_names):
    """AP averaged over an IoU sweep (0.5:0.95 for Car) in 3D and BEV.

    Evaluated on KITTI gt/dt annotations; only the IoU thresholds are swept.

    Args:
        gt_annos: list of KITTI gt anno dicts, one per sample (camera frame)
        dt_annos: list of KITTI dt anno dicts, same length / same order
        class_names: list of class names in the model config

    Returns:
        dict of float, e.g. ``{'ap50_95/Car_3d/moderate': 81.23, ...}``.
        Values follow the official KITTI convention: AP is a percentage
        (0-100), exactly like the ``Car 3D AP`` lines in the eval log.
    """
    from pcdet.datasets.kitti.kitti_object_eval_python.eval import do_eval

    class_ids = _class_ids(class_names)
    if len(class_ids) == 0:
        return {}

    # Same sweep as do_coco_style_eval(), but built here with an explicit int
    # ``num`` because np.linspace(..., num=10.0) raises on numpy>=2.
    n_iou = 10
    min_overlaps = np.zeros([n_iou, 3, len(class_ids)])
    for i, cid in enumerate(class_ids):
        lo, hi, _ = CLASS_TO_RANGE[cid]
        min_overlaps[:, :, i] = np.linspace(lo, hi, n_iou)[:, np.newaxis]

    # compute_aos=False: alpha is not used by 3D/BEV AP
    # do_eval returns 8 arrays (bbox/bev/3d/aos, each with and without R40)
    rets = do_eval(gt_annos, dt_annos, class_ids, min_overlaps, False)
    mAP_bbox, mAP_bev, mAP_3d = rets[0], rets[1], rets[2]
    # do_eval returns [num_class, num_diff, num_minoverlap] -> average the IoU sweep
    mAP_bbox = mAP_bbox.mean(-1)
    mAP_bev = mAP_bev.mean(-1)
    mAP_3d = mAP_3d.mean(-1)

    out = {}
    for j, cid in enumerate(class_ids):
        name = ID_TO_NAME[cid]
        for d, dname in enumerate(DIFFICULTY_NAMES):
            out['ap50_95/%s_3d/%s' % (name, dname)] = float(mAP_3d[j, d])
            out['ap50_95/%s_bev/%s' % (name, dname)] = float(mAP_bev[j, d])
            out['ap50_95/%s_image/%s' % (name, dname)] = float(mAP_bbox[j, d])
    return out


# backwards-compatible alias: the function was previously named compute_coco_map
compute_coco_map = compute_ap_50_95


def compute_prf_3d(gt_annos, dt_annos, class_names, iou_thresh=0.7,
                   score_thresh=0.3, difficulty=1):
    """3D precision / recall / F1 at a fixed IoU and score threshold.

    Matching follows ``compute_statistics_jit`` from the official KITTI eval:
    each gt is matched to the highest-overlap unused detection above
    ``iou_thresh``; detections below ``score_thresh`` are ignored entirely.

    Args:
        difficulty: 0=easy, 1=moderate, 2=hard (KITTI height/occlusion/truncation)

    Returns:
        dict of float, e.g. ``{'prf/Car_3d/precision': 0.91, ...}``
    """
    from pcdet.datasets.kitti.kitti_object_eval_python.eval import (
        calculate_iou_partly, _prepare_data, compute_statistics_jit,
    )

    class_ids = _class_ids(class_names)
    if len(class_ids) == 0:
        return {}

    # NOTE the argument order: the official eval calls it as (dt, gt) so that
    # each returned matrix has shape (num_dt, num_gt), which is what
    # compute_statistics_jit indexes as overlaps[det_idx, gt_idx].
    overlaps, _, _, _ = calculate_iou_partly(dt_annos, gt_annos, 2)

    out = {}
    for cid in class_ids:
        name = ID_TO_NAME[cid]
        (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets,
         dontcares, _, _) = _prepare_data(gt_annos, dt_annos, cid, difficulty)

        tp = fp = fn = 0
        for i in range(len(gt_annos)):
            cur_tp, cur_fp, cur_fn, _, _ = compute_statistics_jit(
                overlaps[i], gt_datas_list[i], dt_datas_list[i],
                ignored_gts[i], ignored_dets[i], dontcares[i],
                2, min_overlap=iou_thresh, thresh=score_thresh,
                compute_fp=True
            )
            tp += int(cur_tp)
            fp += int(cur_fp)
            fn += int(cur_fn)

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        out['prf/%s_3d/precision' % name] = float(precision)
        out['prf/%s_3d/recall' % name] = float(recall)
        out['prf/%s_3d/f1' % name] = float(f1)
        out['prf/%s_3d/tp' % name] = float(tp)
        out['prf/%s_3d/fp' % name] = float(fp)
        out['prf/%s_3d/fn' % name] = float(fn)
    return out


def compute_extra_metrics(gt_annos, dt_annos, class_names, metric_names,
                          logger=None, **kwargs):
    """Run the requested extra metrics, never letting a failure kill training."""
    metric_names = [m.strip().lower() for m in metric_names if m.strip()]
    result = {}

    if len(gt_annos) != len(dt_annos):
        if logger is not None:
            logger.warning(
                'extra metrics skipped: %d gt vs %d detections (distributed eval?)'
                % (len(gt_annos), len(dt_annos))
            )
        return result

    for name in metric_names:
        try:
            # 'coco' is accepted as an alias for old configs -- it only ever meant
            # "COCO-style IoU sweep", never the COCO dataset
            if name in ('ap50_95', 'ap50-95', 'coco'):
                result.update(compute_ap_50_95(gt_annos, dt_annos, class_names))
            elif name == 'prf':
                result.update(compute_prf_3d(
                    gt_annos, dt_annos, class_names,
                    iou_thresh=kwargs.get('iou_thresh', 0.7),
                    score_thresh=kwargs.get('score_thresh', 0.3),
                    difficulty=kwargs.get('difficulty', 1),
                ))
            else:
                if logger is not None:
                    logger.warning('unknown extra metric: %s' % name)
        except Exception as exc:  # noqa: BLE001 - eval must not break training
            if logger is not None:
                logger.warning('extra metric "%s" failed: %r' % (name, exc))
    return result
