# Mean-difference directions

Each NumPy file holds an FP16-trained mean-difference (MD) direction for one model and dataset. The TriviaQA vectors use the grouped training split. Use a transfer vector with `probedir transfer --direction PATH`.

The [probe fits](../probes/models) contain the complete TriviaQA D0–D3 directions and training AUROC profiles. The [geometry panel](../geometry) stores the permutation draws from refitting MD and the residual PCA basis after every swap.
