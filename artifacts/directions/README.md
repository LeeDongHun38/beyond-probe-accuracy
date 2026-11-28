# Mean-difference directions

Each NumPy file contains the fixed FP16 direction for one model and dataset. The five TriviaQA vectors are the camera-ready v2 D0 fits. The transfer and geometry copies for TriviaQA are identical. ARC-Easy and CommonsenseQA retain their original vectors.

Use a transfer vector with `probedir transfer --direction PATH`. The label-permutation experiment refits MD and the residual PCA basis after every swap. Its original draws are in the [geometry panel](../camera_ready/v3/B). The complete TriviaQA D0–D3 safetensors and training AUROC profiles are in the [v2 fits](../camera_ready/v2/models).
