import argparse
import pandas as pd
import numpy as np
import os

import argparse
from filterpy.kalman import KalmanFilter

np.random.seed(0)

def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IUO between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return (o)

def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if (score == None):
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))

class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box. [x1, y1, x2, y2]
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        # self.kf.F是状态变换模型
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1]
            ]
        )
        # self.kf.H是观测函数
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0]
            ]
        )
        # self.kf.R为测量噪声矩阵
        self.kf.R[2:, 2:] *= 10.
        # self.kf.P为协方差矩阵
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        # self.kf.Q为过程噪声矩阵
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if ((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    # 跟踪器数量为0则直接构造结果。
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = iou_batch(detections[:, :4], trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    # 记录未匹配的检测框及轨迹
    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if (iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            # 返回当前边界框估计值。
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if (len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))

def save2Dict(frameNo, bbox, identities=None):
    cur_frame_track_info = []
    for i, box in enumerate(bbox):
        id = int(identities[i]) if identities is not None else 0
        x1, y1, x2, y2 = [int(i) for i in box]
        center_posX = (x1 + x2) / 2.0
        center_posY = (y1 + y2) / 2.0

        cur_frame_track_info.append({
            'frameNo': frameNo,
            'trackid': id,
            'boxesX1': x1,
            'boxesY1': y1,
            'boxesX2': x2,
            'boxesY2': y2,
            'centerX': center_posX,
            'centerY': center_posY
        })
    return cur_frame_track_info

def sort_track(data):

    tracker = Sort(max_age=5, min_hits=4, iou_threshold=0.15)

    # 生成当前track中的所有trackid <class 'numpy.ndarray'>
    unique_frameid = list(data['Frame'].unique())

    all_track_info = []

    for idx_frame in unique_frameid:
        current_data = data[data['Frame'] == idx_frame]
        boxes = current_data[['Upper left corner X', 'Upper left corner Y', 'Lower right corner X', 'Lower right corner Y']].values
        track_result = tracker.update(boxes)
        if track_result.shape[0] != 0:
            bbox_xyxy = track_result[:, :4]
            identities = track_result[:, -1]
            # 当前帧中的所有轨迹信息
            cur_frame_infos = save2Dict(idx_frame, bbox_xyxy, identities)
        else:
            cur_frame_infos = [{
                'frameNo': idx_frame,
                'trackid': -1,
                'boxesX1': -1,
                'boxesY1': -1,
                'boxesX2': -1,
                'boxesY2': -1,
                'centerX': -1,
                'centerY': -1,
            }]
        all_track_info.extend(cur_frame_infos)

    if len(all_track_info) > 0:
        output_df = pd.DataFrame(all_track_info)
        print(f"saved tracking result in {track_file}")
        output_df.to_csv(track_file, index=False, sep=",")
    else:
        print("no track")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # ap.add_argument("-f", "--root_path", default="/home/data/HJZ/zef/exp_pre",
    ap.add_argument("-df", "--detection_file",
                    default="E:\\data\\OCU_ZeF2\\processed\\1_WideType\\20211031_ch12.csv",
                    help="Path to folder")
    ap.add_argument("-tf", "--track_file",
                    default="20211031_ch12.csv",
                    help="Path to folder")
    ap.add_argument("-sf", "--startFrame", default=0, type=int)
    ap.add_argument("-ef", "--endFrame", default=-1, type=int)
    ap.add_argument("-tfloder", "--trafloder", default="E:\\data/OCU_ZeF2/sort_tracker/1_WideType")

    args = vars(ap.parse_args())

    detection_file = args["detection_file"]
    trafloder = args["trafloder"]
    track_file = os.path.join(trafloder, args["track_file"])

    if not os.path.isdir(trafloder):
        os.makedirs(trafloder)

    data = pd.read_csv(detection_file, index_col=None)

    sort_track(data)

