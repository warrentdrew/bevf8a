# !/usr/bin/env python3
"""
Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle3d.apis import manager


class WeightedCrossEntropyLoss(nn.Layer):
    """
    This code is based on https://github.com/TRAILab/CaDDN/blob/5a96b37f16b3c29dd2509507b1cdfdff5d53c558/pcdet/utils/loss_utils.py#L187
    """

    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()

    def forward(self, prediction, target, weights):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predited logits for each class.
            target: (B, #anchors, #classes) float tensor.
                One-hot classification targets.
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted cross entropy loss without reduction
        """
        target = target.argmax(axis=-1)
        loss = F.cross_entropy(prediction, target, reduction='none') * weights
        return loss


@manager.LOSSES.add_component
class WeightedSmoothL1Loss(nn.Layer):
    """
    This code is based on https://github.com/TRAILab/CaDDN/blob/5a96b37f16b3c29dd2509507b1cdfdff5d53c558/pcdet/utils/loss_utils.py#L80

                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    """

    def __init__(self, beta=1.0 / 9.0, code_weights=None):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        self.code_weights = paddle.to_tensor(code_weights)

    @staticmethod
    def smooth_l1_loss(diff, beta):
        """
        as name
        """
        if beta < 1e-5:
            loss = paddle.abs(diff)
        else:
            n_diff = paddle.abs(diff)
            loss = paddle.where(n_diff < beta, 0.5 * n_diff ** 2 / beta,
                                n_diff - 0.5 * beta)

        return loss

    def forward(self, input, target, weights=None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = paddle.where(paddle.isnan(target), input,
                              target)  # ignore nan targets

        diff = input - target
        # code-wise weighting
        if self.code_weights is not None:
            diff = diff * self.code_weights.reshape([1, 1, -1])

        loss = self.smooth_l1_loss(diff, self.beta)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[
                1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)
        return loss

@manager.LOSSES.add_component
class WeightedL1Loss(nn.Layer):
    """
    as name
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedL1Loss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.loss = nn.L1Loss(reduction='none')

    def forward(self, input, target, weight=None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """

        loss = self.loss(input, target)
        if weight is not None:
            loss *= weight

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss * self.loss_weight



# add for bevf
@manager.LOSSES.add_component
class WeightedSigmoidLoss(nn.Layer):
    """Sigmoid cross entropy loss function."""
    def __init__(self, reduction="mean", loss_weight=1.0):
        """Constructor.

        Args:
        gamma: exponent of the modulating factor (1 - p_t) ^ gamma.
        alpha: optional alpha weighting factor to balance positives vs negatives.
        all_zero_negative: bool. if True, will treat all zero as background.
            else, will treat first label as background. only affect alpha.
        """
        super(WeightedSigmoidLoss, self).__init__()
        self._reduction = reduction
        self._loss_weight = loss_weight

    def forward(
        self, prediction_tensor, target_tensor, weights, class_indices=None
    ):
        """Compute loss function.

        Args:
        prediction_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing the predicted logits for each class
        target_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing one-hot encoded classification targets
        weights: a float tensor of shape [batch_size, num_anchors]
        class_indices: (Optional) A 1-D integer tensor of class indices.
            If provided, computes loss only for the specified class indices.

        Returns:
        loss: a float tensor of shape [batch_size, num_anchors, num_classes]
            representing the value of the loss function.
        """
        weights = weights.unsqueeze(2)
        if class_indices is not None:
            weights *= (
                indices_to_dense_vector(class_indices, 
                        prediction_tensor.shape[2]).reshape((1, 1, -1)).cast(prediction_tensor.dtype)
                #.view(1, 1, -1)
                #.type_as(prediction_tensor)
            )
        per_entry_cross_ent = _sigmoid_cross_entropy_with_logits(
            labels=target_tensor, logits=prediction_tensor
        )
        return per_entry_cross_ent * weights


def indices_to_dense_vector(
    indices, size, indices_value=1.0, default_value=0, dtype=np.float32
):
    """Creates dense vector with indices set to specific value and rest to zeros.

    This function exists because it is unclear if it is safe to use
        tf.sparse_to_dense(indices, [size], 1, validate_indices=False)
    with indices which are not ordered.
    This function accepts a dynamic size (e.g. tf.shape(tensor)[0])

    Args:
        indices: 1d Tensor with integer indices which are to be set to
            indices_values.
        size: scalar with size (integer) of output Tensor.
        indices_value: values of elements specified by indices in the output vector
        default_value: values of other elements in the output vector.
        dtype: data type.

    Returns:
        dense 1D Tensor of shape [size] with indices set to indices_values and the
            rest set to default_value.
    """
    dense = paddle.zeros(size).fill_(default_value)
    dense[indices] = indices_value

    return dense


def _sigmoid_cross_entropy_with_logits(logits, labels):
    # to be compatible with tensorflow, we don't use ignore_idx
    loss = paddle.clip(logits, min=0) - logits * labels.cast(logits.dtype)
    loss += paddle.log1p(paddle.exp(-paddle.abs(logits)))
    # transpose_param = [0] + [param[-1]] + param[1:-1]
    # logits = logits.permute(*transpose_param)
    # loss_ftor = nn.NLLLoss(reduce=False)
    # loss = loss_ftor(F.logsigmoid(logits), labels)
    return loss


# add for bevf
@manager.LOSSES.add_component
class WeightedSmoothL1LossIDG(nn.Layer):
    """Smooth L1 localization loss function.

    The smooth L1_loss is defined elementwise as .5 x^2 if |x|<1 and |x|-.5
    otherwise, where x is the difference between predictions and target.

    See also Equation (3) in the Fast R-CNN paper by Ross Girshick (ICCV 2015)
    """

    def __init__(
            self,
            sigma=3.0,
            reduction="mean",
            code_weights=None,
            codewise=True,
            loss_weight=1.0,
            weights_unsqueeze=True):
        super(WeightedSmoothL1LossIDG, self).__init__()
        self._sigma = sigma

        # if code_weights is not None:
        #     self._code_weights = paddle.to_tensor(
        #         code_weights, dtype=paddle.float32)
        # else:
        #     self._code_weights = None
        
        self._code_weights = None

        self._codewise = codewise
        self._reduction = reduction
        self._loss_weight = loss_weight
        		
        self.weights_unsqueeze = weights_unsqueeze

    def __call__(self, prediction_tensor, target_tensor, weights=None):
        """Compute loss function.

        Args:
        prediction_tensor: A float tensor of shape [batch_size, num_anchors,
            code_size] representing the (encoded) predicted locations of objects.
        target_tensor: A float tensor of shape [batch_size, num_anchors,
            code_size] representing the regression targets
        weights: a float tensor of shape [batch_size, num_anchors]

        Returns:
        loss: a float tensor of shape [batch_size, num_anchors] tensor
            representing the value of the loss function.
        """
        diff = prediction_tensor - target_tensor
        if self._code_weights is not None:
            diff = self._code_weights.reshape([1, 1, -1]) * diff
        abs_diff = paddle.abs(diff)
        abs_diff_lt_1 = (abs_diff <= (1 /
                                      (self._sigma ** 2))).cast(abs_diff.dtype)
        loss = abs_diff_lt_1 * 0.5 * paddle.pow(abs_diff * self._sigma, 2) + (
            abs_diff - 0.5 / (self._sigma ** 2)) * (1.0 - abs_diff_lt_1)
        if self._codewise:
            anchorwise_smooth_l1norm = loss
            if weights is not None:
                anchorwise_smooth_l1norm *= weights.unsqueeze(-1) if self.weights_unsqueeze else weights
        else:
            anchorwise_smooth_l1norm = paddle.sum(loss, 2)  #  * weights
            if weights is not None:
                anchorwise_smooth_l1norm *= weights

        return anchorwise_smooth_l1norm


@manager.LOSSES.add_component
class WeightedSoftmaxClassificationLossIDG(nn.Layer):
    """Softmax loss function."""

    def __init__(self, logit_scale=1.0, loss_weight=1.0, name=""):
        """Constructor.

        Args:
        logit_scale: When this value is high, the prediction is "diffused" and
                    when this value is low, the prediction is made peakier.
                    (default 1.0)

        """
        super(WeightedSoftmaxClassificationLossIDG, self).__init__()
        self.name = name
        self._loss_weight = loss_weight
        self._logit_scale = logit_scale

    def __call__(self, prediction_tensor, target_tensor, weights):
        """Compute loss function.

        Args:
        prediction_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing the predicted logits for each class
        target_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing one-hot encoded classification targets
        weights: a float tensor of shape [batch_size, num_anchors]

        Returns:
        loss: a float tensor of shape [batch_size, num_anchors]
            representing the value of the loss function.
        """
        num_classes = prediction_tensor.shape[-1]
        prediction_tensor = prediction_tensor / self._logit_scale
        per_row_cross_ent = _softmax_cross_entropy_with_logits(
            labels=target_tensor.reshape([-1, num_classes]),
            logits=prediction_tensor.reshape([-1, num_classes]), )

        return per_row_cross_ent.reshape(weights.shape) * weights

def _softmax_cross_entropy_with_logits(logits, labels):
    """Softmax cross entropy with logits."""
    param = list(range(len(logits.shape)))
    transpose_param = [0] + [param[-1]] + param[1:-1]
    logits = logits.transpose(transpose_param)  # [N, ..., C] -> [N, C, ...]
    loss_ftor = nn.CrossEntropyLoss(reduction="none")
    loss = loss_ftor(logits, labels.argmax(axis=-1))
    return loss
    