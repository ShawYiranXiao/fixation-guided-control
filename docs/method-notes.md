# Method notes

DeepGazeIII predicts the next fixation conditioned on an image and fixation history. In the archived conditioning pipeline, four locations are sampled uniformly as a synthetic history and multiple next-fixation log-density maps are generated for each image. These maps are model predictions under artificial histories, rather than measured human scanpaths.

## Fusion definitions

`fusion_mean` averages log densities. Exponentiating and renormalizing that result yields geometric pooling; it is different from averaging normalized probability distributions. `fusion_softor` computes `1 - product(1 - p)` after normalizing each map. This score is not automatically a probability density and does not establish independence between predictions. `fusion_weighted_softmax` is the historical name for a maximum-based weighting rule; its weights are not calculated with softmax.

These definitions are preserved in the publication copies. Some exploration names are aliases of the same computation; for example, the softmax-lighten and peak-preserve variants both use a pixel-wise maximum. Do not treat them as independent methods in an evaluation.

## Display and cache behavior

The original conditions use per-image min-max scaling followed by a jet color map. This preserves relative spatial structure but discards a common absolute scale. Different model, image, history and center-bias configurations must use separate cache directories because the historical cache identity is based on the image key. The release rejects unsafe filename keys; it does not change the research cache format or fusion semantics.

Setting a uniform explicit center bias does not remove all bias in the model or its training data. The separate `simulated_scanpath.py` samples successive predictions, but its timestamp intervals are manually specified and are not a learned fixation-duration model.

## Evaluation scope

The master's work includes qualitative generation and fusion comparisons. A claim of control over human-perceived complexity requires paired conditions, fixed captions and seeds, independent ratings and an appropriate eye-tracking study. The release tests check software regressions and operator behavior, not perceptual effectiveness.
