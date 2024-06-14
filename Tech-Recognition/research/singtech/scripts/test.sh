CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 240221-te-01 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingTechRecog/checkpoints/240221-te-01/model_ckpt_best.pt"
#overall |     auroc: 0.847
#overall | precision: 0.282
#overall |    recall: 0.235
#overall |        f1: 0.241
#overall |  accuracy: 0.928
#------------------------------------------------------------
#mix     |     auroc: 0.691 || falsetto |     auroc: 0.777 || breathe |     auroc: 0.651
#mix     | precision: 0.056 || falsetto | precision: 0.093 || breathe | precision: 0.029
#mix     |    recall: 0.036 || falsetto |    recall: 0.078 || breathe |    recall: 0.018
#mix     |        f1: 0.037 || falsetto |        f1: 0.083 || breathe |        f1: 0.020
#mix     |  accuracy: 0.883 || falsetto |  accuracy: 0.848 || breathe |  accuracy: 0.959
#------------------------------------------------------------
#bubble  |     auroc: 0.860 || strong   |     auroc: 0.692 || weak    |     auroc: 0.775
#bubble  | precision: 0.036 || strong   | precision: 0.047 || weak    | precision: 0.070
#bubble  |    recall: 0.009 || strong   |    recall: 0.027 || weak    |    recall: 0.071
#bubble  |        f1: 0.013 || strong   |        f1: 0.031 || weak    |        f1: 0.070
#bubble  |  accuracy: 0.977 || strong   |  accuracy: 0.907 || weak    |  accuracy: 0.993

# 把tgt为0的p,r,f给去掉之后
#overall |     auroc: 0.847
#overall | precision: 0.491
#overall |    recall: 0.409
#overall |        f1: 0.420
#overall |  accuracy: 0.928
#------------------------------------------------------------
#mix     |     auroc: 0.691 || falsetto |     auroc: 0.777 || breathe |     auroc: 0.651
#mix     | precision: 0.358 || falsetto | precision: 0.737 || breathe | precision: 0.367
#mix     |    recall: 0.229 || falsetto |    recall: 0.615 || breathe |    recall: 0.229
#mix     |        f1: 0.232 || falsetto |        f1: 0.660 || breathe |        f1: 0.254
#mix     |  accuracy: 0.883 || falsetto |  accuracy: 0.848 || breathe |  accuracy: 0.959
#------------------------------------------------------------
#bubble  |     auroc: 0.860 || strong   |     auroc: 0.692 || weak    |     auroc: 0.775
#bubble  | precision: 0.352 || strong   | precision: 0.667 || weak    | precision: 0.987
#bubble  |    recall: 0.086 || strong   |    recall: 0.381 || weak    |    recall: 1.000
#bubble  |        f1: 0.126 || strong   |        f1: 0.434 || weak    |        f1: 0.993
#bubble  |  accuracy: 0.977 || strong   |  accuracy: 0.907 || weak    |  accuracy: 0.993

CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 240221-te-02 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingTechRecog/checkpoints/240221-te-02/model_ckpt_best.pt"
#overall |     auroc: 0.839
#overall | precision: 0.587
#overall |    recall: 0.512
#overall |        f1: 0.529
#overall |  accuracy: 0.935
#------------------------------------------------------------
#mix     |     auroc: 0.701 || falsetto |     auroc: 0.812 || breathe |     auroc: 0.623
#mix     | precision: 0.313 || falsetto | precision: 1.000 || breathe | precision: 0.322
#mix     |    recall: 0.179 || falsetto |    recall: 0.994 || breathe |    recall: 0.307
#mix     |        f1: 0.211 || falsetto |        f1: 0.997 || breathe |        f1: 0.314
#mix     |  accuracy: 0.899 || falsetto |  accuracy: 0.887 || breathe |  accuracy: 0.966
#------------------------------------------------------------
#bubble  |     auroc: 0.854 || strong   |     auroc: 0.778 || weak    |     auroc: 0.773
#bubble  | precision: 0.435 || strong   | precision: 0.778 || weak    | precision: 0.886
#bubble  |    recall: 0.233 || strong   |    recall: 0.561 || weak    |    recall: 0.806
#bubble  |        f1: 0.285 || strong   |        f1: 0.604 || weak    |        f1: 0.822
#bubble  |  accuracy: 0.981 || strong   |  accuracy: 0.893 || weak    |  accuracy: 0.987

CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 240221-te-03 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingTechRecog/checkpoints/240221-te-03/model_ckpt_best.pt"
#overall |     auroc: 0.842
#overall | precision: 0.526
#overall |    recall: 0.346
#overall |        f1: 0.386
#overall |  accuracy: 0.937
#--------------------------------------------------------------------------------
#mix     |     auroc: 0.685 || falsetto |     auroc: 0.812 || breathe |     auroc: 0.730
#mix     | precision: 0.350 || falsetto | precision: 0.750 || breathe | precision: 0.325
#mix     |    recall: 0.102 || falsetto |    recall: 0.451 || breathe |    recall: 0.148
#mix     |        f1: 0.128 || falsetto |        f1: 0.523 || breathe |        f1: 0.170
#mix     |  accuracy: 0.899 || falsetto |  accuracy: 0.870 || breathe |  accuracy: 0.955
#--------------------------------------------------------------------------------
#bubble  |     auroc: 0.824 || strong   |     auroc: 0.773 || weak    |     auroc: 0.747
#bubble  | precision: 0.441 || strong   | precision: 0.667 || weak    | precision: 0.997
#bubble  |    recall: 0.221 || strong   |    recall: 0.372 || weak    |    recall: 0.944
#bubble  |        f1: 0.273 || strong   |        f1: 0.430 || weak    |        f1: 0.961
#bubble  |  accuracy: 0.981 || strong   |  accuracy: 0.933 || weak    |  accuracy: 0.984

CUDA_VISIBLE_DEVICES=1 python tasks/run.py --exp_name 240221-te-04 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingTechRecog/checkpoints/240221-te-04/model_ckpt_best.pt"
#overall |     auroc: 0.874
#overall | precision: 0.612
#overall |    recall: 0.478
#overall |        f1: 0.512
#overall |  accuracy: 0.947
#--------------------------------------------------------------------------------
#mix     |     auroc: 0.708 || falsetto |     auroc: 0.812 || breathe |     auroc: 0.799
#mix     | precision: 0.283 || falsetto | precision: 0.812 || breathe | precision: 0.512
#mix     |    recall: 0.253 || falsetto |    recall: 0.516 || breathe |    recall: 0.431
#mix     |        f1: 0.260 || falsetto |        f1: 0.603 || breathe |        f1: 0.450
#mix     |  accuracy: 0.903 || falsetto |  accuracy: 0.882 || breathe |  accuracy: 0.971
#--------------------------------------------------------------------------------
#bubble  |     auroc: 0.883 || strong   |     auroc: 0.778 || weak    |     auroc: 0.768
#bubble  | precision: 0.408 || strong   | precision: 0.889 || weak    | precision: 0.997
#bubble  |    recall: 0.180 || strong   |    recall: 0.858 || weak    |    recall: 0.875
#bubble  |        f1: 0.248 || strong   |        f1: 0.872 || weak    |        f1: 0.889
#bubble  |  accuracy: 0.981 || strong   |  accuracy: 0.959 || weak    |  accuracy: 0.988

