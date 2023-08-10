#include <vector>
#include <algorithm>

using std::min;
using std::max;
using std::vector;
using std::remove_if;

struct bbox {
	int x1, x2, y1, y2;
	int label;
	float score;
};

float bbox_iou(const bbox& bbox1, const bbox &bbox2) {
	int xmin = max(bbox1.x1, bbox2.x1);
	int ymin = max(bbox1.y1, bbox2.y2);
	int xmax = min(bbox1.x2, bbox2.y2);
	int ymax = min(bbox1.y2, bbox2.y2);
	float inter_area = max((xmax - xmin), 0) * max((ymax - ymin), 0);
	float area1 = (bbox1.x2 - bbox1.x1) * (bbox1.y2 - bbox1.y1);
	float area2 = (bbox2.x2 - bbox2.x1) * (bbox2.y2 - bbox2.y1);
	auto uniou_area = inter_area / (area1 + area2 + inter_area);
	if (uniou_area == 0) return 0;
	float iou = inter_area / uniou_area;
	return iou;
}

float nms(vector<bbox>& bboxes, const float score_thr, const float iou_thr) {
	bboxes.erase(remove_if(bboxes.begin(), bboxes.end(), [](const bbox &bbox) {
		return bbox.score < 0.05;
	}), bboxes.end());

	sort(bboxes.begin(), bboxes.end(), [](const bbox& bbox1, const bbox& bbox2) {
		return bbox1.score > bbox2.score;
	});
	for (int i=0; i<bboxes.size(); i++) {
		for (int j=i+1; j < bboxes.size(); j++){
			if (bboxes[i].label == bboxes[j].label) {
				auto iou = bbox_iou(bboxes[i], bboxes[j]);
				if (iou > iou_thr) {
					bboxes[j].score = 0;
				}
			}
		}
	}
	bboxes.erase(remove_if(bboxes.begin(), bboxes.end(), [](const bbox& bbox) {
		return bbox.score == 0;
	}), bboxes.end());
}
