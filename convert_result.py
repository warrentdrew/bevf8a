import paddle
import pickle


results = paddle.load("test_results_exp5_dev_1031.pdt")

for result in results:
    result["pts_bbox"]['boxes_3d'] = result["pts_bbox"]['boxes_3d'].numpy()
    result["pts_bbox"]['labels_3d'] = result["pts_bbox"]['labels_3d'].numpy() # .cpu().numpy()
    result["pts_bbox"]['scores_3d'] = result["pts_bbox"]['scores_3d'].numpy() #.cpu().numpy()
    result["pts_bbox"]["combo_uid"] = result["pts_bbox"]["combo_uid"].numpy()

with open('test_results_exp5_dev_1031_convert.pkl', 'wb') as f:
    pickle.dump(results, f)