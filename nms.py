import numpy as np

def bbox_iou(bbox1:np.ndarray, bbox2:np.ndarray) -> float:
    xmin = max(bbox1[0], bbox2[0])
    xmax = min(bbox1[2], bbox2[2])
    ymin = max(bbox1[1], bbox2[1])
    ymax = min(bbox1[3], bbox2[3])
    inter_area = (xmax - xmin) * (ymax - ymin)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - inter_area
    return inter_area / (union + 1e-6)

def nms(bboxes:np.ndarray, scores:np.ndarray, score_thr:float, iou_thr:float) -> np.ndarray:
    valid = scores > score_thr
    bboxes = bboxes[valid]
    scores = scores[valid]
    inds = scores.argsort()
    inds = inds[::-1]
    keep_inds = []
    while inds:
        cur = inds[0]
        cur_score = scores[cur]
        cur_bbox = bboxes[cur]
        keep = True 
        for ind in keep_inds:
            bbox = bboxes[ind]
            iou = bbox_iou(bbox, cur_bbox)
            if iou > iou_thr:
                keep = False
                break
            if keep:
                keep_inds.append(cur)
            inds = inds[1:]
    return np.array(keep_inds)