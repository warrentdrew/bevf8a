import paddle
from paddle3d.ops import iou3d_idg


boxes = paddle.load('ptbc_corners.pdt')
confidence_map = paddle.load('confidence_map.pdt')
print(boxes.shape)
print(confidence_map.shape)
# boxes = boxes.reshape((-1, 4, 2))
# confidence_map = confidence_map.reshape((-1, 1))
# print(boxes.min(), boxes.max())
# print(confidence_map.sum())
confidences = iou3d_idg.parsing_to_boxes_confidence(boxes, confidence_map)
# confidences2 = confidences.clone()
print(confidences.min())
conf = confidences.index_select(paddle.to_tensor([0, 1, 3, 4, 6]), axis = 0) #[:10]
# print(confidences.shape)