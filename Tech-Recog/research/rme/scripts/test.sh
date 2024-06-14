CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 231114-me2-05 --infer
python research/rme/evaluation.py --generated-dir /mnt/sdb/liruiqi/SingingDictation/checkpoints/231114-me2-05/generated_80000_/midi
#onset  |   precision: 0.803  ||  offset |     precision: 0.738
#onset  |      recall: 0.700  ||  offset |        recall: 0.646
#onset  |          f1: 0.738  ||  offset |            f1: 0.679
#melody |          VR: 0.982  ||  pitch  |     precision: 0.419
#melody |         VFA: 0.202  ||  pitch  |        recall: 0.374
#melody |         RPA: 0.733  ||  pitch  |            f1: 0.391
#melody |         RCA: 0.734  ||  pitch  | overlap_ratio: 0.800
#melody |          OA: 0.742

# 把evaluation集成到了test里
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 231115-me2-02 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231115-me2-02/model_ckpt_best.pt"
#onset  |   precision: 0.807  ||  offset |     precision: 0.769
#onset  |      recall: 0.730  ||  offset |        recall: 0.697
#onset  |          f1: 0.757  ||  offset |            f1: 0.722
#melody |          VR: 0.988  ||  pitch  |     precision: 0.465
#melody |         VFA: 0.284  ||  pitch  |        recall: 0.430
#melody |         RPA: 0.768  ||  pitch  |            f1: 0.443
#melody |         RCA: 0.768  ||  pitch  | overlap_ratio: 0.800
#melody |          OA: 0.764

# *
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 231115-me2-03 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231115-me2-03/model_ckpt_best_26000.pt"
#onset  |   precision: 0.838  ||  offset |     precision: 0.783
#onset  |      recall: 0.711  ||  offset |        recall: 0.668
#onset  |          f1: 0.760  ||  offset |            f1: 0.712
#melody |          VR: 0.987  ||  pitch  |     precision: 0.474
#melody |         VFA: 0.241  ||  pitch  |        recall: 0.416
#melody |         RPA: 0.762  ||  pitch  |            f1: 0.439
#melody |         RCA: 0.762  ||  pitch  | overlap_ratio: 0.812
#melody |          OA: 0.762

CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 231116-me2-02 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231116-me2-02/model_ckpt_best_39000.pt"
#onset  |   precision: 0.740  ||  offset |     precision: 0.630
#onset  |      recall: 0.634  ||  offset |        recall: 0.547
#onset  |          f1: 0.675  ||  offset |            f1: 0.578
#melody |          VR: 0.981  ||  pitch  |     precision: 0.342
#melody |         VFA: 0.369  ||  pitch  |        recall: 0.303
#melody |         RPA: 0.739  ||  pitch  |            f1: 0.318
#melody |         RCA: 0.740  ||  pitch  | overlap_ratio: 0.738
#melody |          OA: 0.731

CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 231116-me2-03 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231116-me2-03/model_ckpt_best_17000.pt"
#onset  |   precision: 0.815  ||  offset |     precision: 0.766
#onset  |      recall: 0.726  ||  offset |        recall: 0.683
#onset  |          f1: 0.758  ||  offset |            f1: 0.712
#melody |          VR: 0.985  ||  pitch  |     precision: 0.457
#melody |         VFA: 0.235  ||  pitch  |        recall: 0.415
#melody |         RPA: 0.758  ||  pitch  |            f1: 0.431
#melody |         RCA: 0.758  ||  pitch  | overlap_ratio: 0.808
#melody |          OA: 0.760

CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 231116-me2-04 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231116-me2-04/model_ckpt_best.pt"
#onset  |   precision: 0.814  ||  offset |     precision: 0.755
#onset  |      recall: 0.750  ||  offset |        recall: 0.698
#onset  |          f1: 0.772  ||  offset |            f1: 0.717
#melody |          VR: 0.985  ||  pitch  |     precision: 0.469
#melody |         VFA: 0.285  ||  pitch  |        recall: 0.439
#melody |         RPA: 0.766  ||  pitch  |            f1: 0.449
#melody |         RCA: 0.766  ||  pitch  | overlap_ratio: 0.814
#melody |          OA: 0.761

CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 231116-me2-05 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231116-me2-05/model_ckpt_best.pt"
#onset  |   precision: 0.824  ||  offset |     precision: 0.768
#onset  |      recall: 0.720  ||  offset |        recall: 0.674
#onset  |          f1: 0.758  ||  offset |            f1: 0.708
#melody |          VR: 0.958  ||  pitch  |     precision: 0.473
#melody |         VFA: 0.261  ||  pitch  |        recall: 0.421
#melody |         RPA: 0.740  ||  pitch  |            f1: 0.441
#melody |         RCA: 0.741  ||  pitch  | overlap_ratio: 0.808
#melody |          OA: 0.741

CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 231116-me2-06 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231116-me2-06/model_ckpt_best.pt"
#onset  |   precision: 0.826  ||  offset |     precision: 0.770
#onset  |      recall: 0.688  ||  offset |        recall: 0.643
#onset  |          f1: 0.740  ||  offset |            f1: 0.690
#melody |          VR: 0.975  ||  pitch  |     precision: 0.449
#melody |         VFA: 0.279  ||  pitch  |        recall: 0.385
#melody |         RPA: 0.736  ||  pitch  |            f1: 0.410
#melody |         RCA: 0.736  ||  pitch  | overlap_ratio: 0.798
#melody |          OA: 0.734

# *
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 231116-me2-07 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231116-me2-07/model_ckpt_best.pt"
#onset  |   precision: 0.808  ||  offset |     precision: 0.765
#onset  |      recall: 0.761  ||  offset |        recall: 0.721
#onset  |*         f1: 0.776  ||  offset |*           f1: 0.735
#melody |          VR: 0.986  ||  pitch  |     precision: 0.482
#melody |         VFA: 0.292  ||  pitch  |        recall: 0.458
#melody |         RPA: 0.779  ||  pitch  |            f1: 0.466
#melody |         RCA: 0.779  ||  pitch  | overlap_ratio: 0.811
#melody |*         OA: 0.771

CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 231116-me2-13 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231116-me2-13/model_ckpt_best.pt"
#onset  |   precision: 0.819  ||  offset |     precision: 0.761
#onset  |      recall: 0.737  ||  offset |        recall: 0.687
#onset  |*         f1: 0.767  ||  offset |*           f1: 0.714
#melody |          VR: 0.986  ||  pitch  |     precision: 0.470
#melody |         VFA: 0.281  ||  pitch  |        recall: 0.430
#melody |         RPA: 0.769  ||  pitch  |            f1: 0.445
#melody |         RCA: 0.769  ||  pitch  | overlap_ratio: 0.805
#melody |*         OA: 0.764

CUDA_VISIBLE_DEVICES=1 python tasks/run.py --exp_name 231116-me2-14 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231116-me2-14/model_ckpt_best.pt"
#onset  |   precision: 0.783  ||  offset |     precision: 0.662
#onset  |      recall: 0.742  ||  offset |        recall: 0.634
#onset  |*         f1: 0.755  ||  offset |*           f1: 0.642
#melody |          VR: 0.985  ||  pitch  |     precision: 0.410
#melody |         VFA: 0.345  ||  pitch  |        recall: 0.393
#melody |         RPA: 0.771  ||  pitch  |            f1: 0.399
#melody |         RCA: 0.771  ||  pitch  | overlap_ratio: 0.773
#melody |*         OA: 0.760

CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 231120-me2-01 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231120-me2-01/model_ckpt_best.pt"
#onset  |   precision: 0.905  ||  offset |     precision: 0.899
#onset  |      recall: 0.946  ||  offset |        recall: 0.940
#onset  |*         f1: 0.920  ||  offset |*           f1: 0.914
#melody |          VR: 0.993  ||  pitch  |     precision: 0.697
#melody |         VFA: 0.037  ||  pitch  |        recall: 0.723
#melody |*        RPA: 0.831  ||  pitch  |            f1: 0.707
#melody |         RCA: 0.831  ||  pitch  | overlap_ratio: 0.886
#melody |*         OA: 0.844

CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 231120-me2-02 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231120-me2-02/model_ckpt_best.pt"
#onset  |   precision: 0.900  ||  offset |     precision: 0.894
#onset  |      recall: 0.951  ||  offset |        recall: 0.944
#onset  |*         f1: 0.920  ||  offset |*           f1: 0.914
#melody |          VR: 0.992  ||  pitch  |     precision: 0.697
#melody |         VFA: 0.041  ||  pitch  |        recall: 0.731
#melody |*        RPA: 0.834  ||  pitch  |            f1: 0.711
#melody |         RCA: 0.834  ||  pitch  | overlap_ratio: 0.881
#melody |*         OA: 0.846

# *
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 231120-me2-03 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231120-me2-03/model_ckpt_best.pt"
#onset  |   precision: 0.918  ||  offset |     precision: 0.915
#onset  |      recall: 0.940  ||  offset |        recall: 0.936
#onset  |*         f1: 0.924  ||  offset |*           f1: 0.921
#melody |          VR: 0.990  ||  pitch  |     precision: 0.720
#melody |         VFA: 0.033  ||  pitch  |        recall: 0.732
#melody |*        RPA: 0.838  ||  pitch  |            f1: 0.723
#melody |         RCA: 0.838  ||  pitch  | overlap_ratio: 0.904
#melody |*         OA: 0.851

CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 231120-me2-04 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231120-me2-04/model_ckpt_best.pt"
#onset  |   precision: 0.920  ||  offset |     precision: 0.916
#onset  |      recall: 0.940  ||  offset |        recall: 0.935
#onset  |*         f1: 0.925  ||  offset |*           f1: 0.921
#melody |          VR: 0.993  ||  pitch  |     precision: 0.715
#melody |         VFA: 0.035  ||  pitch  |        recall: 0.729
#melody |         RPA: 0.836  ||  pitch  |            f1: 0.719
#melody |         RCA: 0.836  ||  pitch  | overlap_ratio: 0.899
#melody |*         OA: 0.849

############################### 第二部分 ###############################

CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 231122-me3-01 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231122-me3-01/model_ckpt_best.pt"
#onset  |   precision: 0.916  ||  offset |     precision: 0.905
#onset  |      recall: 0.955  ||  offset |        recall: 0.944
#onset  |*         f1: 0.931  ||  offset |*           f1: 0.920
#melody |          VR: 0.991  ||  pitch  |     precision: 0.710
#melody |         VFA: 0.048  ||  pitch  |        recall: 0.735
#melody |*        RPA: 0.818  ||  pitch  |            f1: 0.719
#melody |         RCA: 0.818  ||  pitch  | overlap_ratio: 0.850
#melody |*         OA: 0.832

CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 231122-me3-02 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231122-me3-02/model_ckpt_best.pt"
#onset  |   precision: 0.937  ||  offset |     precision: 0.930
#onset  |      recall: 0.949  ||  offset |        recall: 0.942
#onset  |*         f1: 0.939  ||  offset |*           f1: 0.932
#melody |          VR: 0.992  ||  pitch  |     precision: 0.745
#melody |         VFA: 0.047  ||  pitch  |        recall: 0.754
#melody |*        RPA: 0.829  ||  pitch  |            f1: 0.747
#melody |         RCA: 0.829  ||  pitch  | overlap_ratio: 0.854
#melody |*         OA: 0.842

CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 231123-me5-01 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231123-me5-01/model_ckpt_best.pt"
#onset  |   precision: 0.934  ||  offset |     precision: 0.927
#onset  |      recall: 0.954  ||  offset |        recall: 0.948
#onset  |*         f1: 0.940  ||  offset |*           f1: 0.933
#melody |          VR: 0.992  ||  pitch  |     precision: 0.746
#melody |         VFA: 0.047  ||  pitch  |        recall: 0.761
#melody |*        RPA: 0.831  ||  pitch  |            f1: 0.751
#melody |         RCA: 0.832  ||  pitch  | overlap_ratio: 0.858
#melody |*         OA: 0.844

CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 231123-me5-02 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231123-me5-02/model_ckpt_best.pt"
#onset  |   precision: 0.927  ||  offset |     precision: 0.922
#onset  |      recall: 0.965  ||  offset |        recall: 0.960
#onset  |*         f1: 0.942  ||  offset |*           f1: 0.937
#melody |          VR: 0.995  ||  pitch  |     precision: 0.746
#melody |         VFA: 0.032  ||  pitch  |        recall: 0.772
#melody |*        RPA: 0.843  ||  pitch  |            f1: 0.757
#melody |         RCA: 0.845  ||  pitch  | overlap_ratio: 0.894
#melody |*         OA: 0.856

# 同231123-me5-02，但是加了 word_bd 的regulate 之后！note_bd_ref_min_gap=40
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 231123-me5-02 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231123-me5-02/model_ckpt_best.pt,note_bd_ref_min_gap=40"
#onset  |   precision: 0.927  ||  offset |     precision: 0.923
#onset  |      recall: 0.968  ||  offset |        recall: 0.964
#onset  |*         f1: 0.943  ||  offset |*           f1: 0.939
#melody |          VR: 0.997  ||  pitch  |     precision: 0.749
#melody |         VFA: 0.006  ||  pitch  |        recall: 0.777
#melody |*        RPA: 0.860  ||  pitch  |            f1: 0.760
#melody |         RCA: 0.862  ||  pitch  | overlap_ratio: 0.958
#melody |*         OA: 0.873

# 同231123-me5-02，但是加了 word_bd 的regulate 之后！note_bd_ref_min_gap=30
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 231123-me5-02 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231123-me5-02/model_ckpt_best.pt,note_bd_ref_min_gap=30"
#onset  |   precision: 0.926  ||  offset |     precision: 0.922
#onset  |      recall: 0.968  ||  offset |        recall: 0.964
#onset  |*         f1: 0.943  ||  offset |*           f1: 0.938
#melody |          VR: 0.996  ||  pitch  |     precision: 0.748
#melody |         VFA: 0.004  ||  pitch  |        recall: 0.777
#melody |*        RPA: 0.859  ||  pitch  |            f1: 0.760
#melody |         RCA: 0.861  ||  pitch  | overlap_ratio: 0.957
#melody |*         OA: 0.873

# *
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 231123-me5-03 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231123-me5-03/model_ckpt_best.pt"
#onset  |   precision: 0.920  ||  offset |     precision: 0.915
#onset  |      recall: 0.971  ||  offset |        recall: 0.967
#onset  |*         f1: 0.941  ||  offset |*           f1: 0.937
#melody |          VR: 0.998  ||  pitch  |     precision: 0.744
#melody |         VFA: 0.008  ||  pitch  |        recall: 0.778
#melody |*        RPA: 0.864  ||  pitch  |            f1: 0.758
#melody |         RCA: 0.864  ||  pitch  | overlap_ratio: 0.959
#melody |*         OA: 0.876

# *
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 231123-me5-04 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231123-me5-04/model_ckpt_best.pt"
#onset  |   precision: 0.927  ||  offset |     precision: 0.923
#onset  |      recall: 0.966  ||  offset |        recall: 0.962
#onset  |*         f1: 0.942  ||  offset |*           f1: 0.938
#melody |          VR: 0.998  ||  pitch  |     precision: 0.749
#melody |         VFA: 0.007  ||  pitch  |        recall: 0.776
#melody |*        RPA: 0.863  ||  pitch  |            f1: 0.760
#melody |         RCA: 0.863  ||  pitch  | overlap_ratio: 0.961
#melody |*         OA: 0.875

CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 231124-me5-01 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231124-me5-01/model_ckpt_best.pt"
#onset  |   precision: 1.000  ||  offset |     precision: 0.999
#onset  |      recall: 0.800  ||  offset |        recall: 0.800
#onset  |*         f1: 0.881  ||  offset |*           f1: 0.881
#melody |          VR: 0.998  ||  pitch  |     precision: 0.672
#melody |         VFA: 0.014  ||  pitch  |        recall: 0.563
#melody |*        RPA: 0.793  ||  pitch  |            f1: 0.609
#melody |         RCA: 0.795  ||  pitch  | overlap_ratio: 0.946
#melody |*         OA: 0.811

CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 231212-me5-02 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231212-me5-02/model_ckpt_best.pt"
#onset  |   precision: 0.994  ||  offset |     precision: 0.994
#onset  |      recall: 0.734  ||  offset |        recall: 0.734
#onset  |*         f1: 0.833  ||  offset |*           f1: 0.832
#melody |          VR: 0.926  ||  pitch  |     precision: 0.494
#melody |         VFA: 0.021  ||  pitch  |        recall: 0.383
#melody |*        RPA: 0.491  ||  pitch  |            f1: 0.427
#melody |         RCA: 0.505  ||  pitch  | overlap_ratio: 0.889
#melody |*         OA: 0.537

CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 231213-me5-01 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231213-me5-01/model_ckpt_best.pt"
#onset  |   precision: 0.936  ||  offset |     precision: 0.932
#onset  |      recall: 0.954  ||  offset |        recall: 0.950
#onset  |*         f1: 0.941  ||  offset |*           f1: 0.937
#melody |          VR: 0.997  ||  pitch  |     precision: 0.749
#melody |         VFA: 0.006  ||  pitch  |        recall: 0.763
#melody |*        RPA: 0.857  ||  pitch  |            f1: 0.754
#melody |         RCA: 0.857  ||  pitch  | overlap_ratio: 0.961
#melody |*         OA: 0.870

CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 231213-me5-02 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231213-me5-02/model_ckpt_best.pt"
#onset  |   precision: 0.910  ||  offset |     precision: 0.905
#onset  |      recall: 0.975  ||  offset |        recall: 0.969
#onset  |*         f1: 0.937  ||  offset |*           f1: 0.932
#melody |          VR: 0.997  ||  pitch  |     precision: 0.732
#melody |         VFA: 0.007  ||  pitch  |        recall: 0.775
#melody |*        RPA: 0.862  ||  pitch  |            f1: 0.750
#melody |         RCA: 0.862  ||  pitch  | overlap_ratio: 0.954
#melody |*         OA: 0.875

# 过拟合很严重
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 231214-me6-01 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231214-me6-01/model_ckpt_best.pt"
# onset  |   precision: 0.950  ||  offset |     precision: 0.944
#onset  |      recall: 0.906  ||  offset |        recall: 0.901
#onset  |*         f1: 0.923  ||  offset |*           f1: 0.918
#melody |          VR: 0.998  ||  pitch  |     precision: 0.701
#melody |         VFA: 0.008  ||  pitch  |        recall: 0.676
#melody |*        RPA: 0.820  ||  pitch  |            f1: 0.686
#melody |         RCA: 0.823  ||  pitch  | overlap_ratio: 0.964
#melody |*         OA: 0.836

CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 231214-me6-02 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231214-me6-02/model_ckpt_best.pt"
#onset  |   precision: 0.934  ||  offset |     precision: 0.929
#onset  |      recall: 0.918  ||  offset |        recall: 0.913
#onset  |*         f1: 0.921  ||  offset |*           f1: 0.916
#melody |          VR: 0.998  ||  pitch  |     precision: 0.695
#melody |         VFA: 0.006  ||  pitch  |        recall: 0.687
#melody |*        RPA: 0.829  ||  pitch  |            f1: 0.688
#melody |         RCA: 0.829  ||  pitch  | overlap_ratio: 0.962
#melody |*         OA: 0.844

CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 231214-me6-03 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231214-me6-03/model_ckpt_best.pt"
#onset  |   precision: 0.936  ||  offset |     precision: 0.930
#onset  |      recall: 0.909  ||  offset |        recall: 0.903
#onset  |*         f1: 0.917  ||  offset |*           f1: 0.911
#melody |          VR: 0.997  ||  pitch  |     precision: 0.680
#melody |         VFA: 0.007  ||  pitch  |        recall: 0.666
#melody |*        RPA: 0.813  ||  pitch  |            f1: 0.670
#melody |         RCA: 0.815  ||  pitch  | overlap_ratio: 0.956
#melody |*         OA: 0.830

CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 231217-me7-01 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231217-me7-01/model_ckpt_best.pt"
#onset  |   precision: 0.948  ||  offset |     precision: 0.943
#onset  |      recall: 0.926  ||  offset |        recall: 0.921
#onset  |*         f1: 0.933  ||  offset |*           f1: 0.928
#melody |          VR: 0.997  ||  pitch  |     precision: 0.724
#melody |         VFA: 0.006  ||  pitch  |        recall: 0.710
#melody |*        RPA: 0.839  ||  pitch  |            f1: 0.715
#melody |         RCA: 0.839  ||  pitch  | overlap_ratio: 0.965
#melody |*         OA: 0.854

CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 231217-me7-03 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231217-me7-03/model_ckpt_best.pt"
#onset  |   precision: 0.955  ||  offset |     precision: 0.949
#onset  |      recall: 0.922  ||  offset |        recall: 0.916
#onset  |*         f1: 0.935  ||  offset |*           f1: 0.928
#melody |          VR: 0.997  ||  pitch  |     precision: 0.736
#melody |         VFA: 0.006  ||  pitch  |        recall: 0.714
#melody |*        RPA: 0.843  ||  pitch  |            f1: 0.723
#melody |         RCA: 0.843  ||  pitch  | overlap_ratio: 0.964
#melody |*         OA: 0.857

CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 231217-me7-04 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231217-me7-04/model_ckpt_best.pt"
#onset  |   precision: 0.945  ||  offset |     precision: 0.939
#onset  |      recall: 0.931  ||  offset |        recall: 0.925
#onset  |*         f1: 0.934  ||  offset |*           f1: 0.928
#melody |          VR: 0.997  ||  pitch  |     precision: 0.723
#melody |         VFA: 0.005  ||  pitch  |        recall: 0.714
#melody |*        RPA: 0.841  ||  pitch  |            f1: 0.716
#melody |         RCA: 0.842  ||  pitch  | overlap_ratio: 0.964
#melody |*         OA: 0.856

CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 231217-me7-05 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231217-me7-05/model_ckpt_best.pt"
#onset  |   precision: 0.945  ||  offset |     precision: 0.938
#onset  |      recall: 0.935  ||  offset |        recall: 0.928
#onset  |*         f1: 0.936  ||  offset |*           f1: 0.929
#melody |          VR: 0.996  ||  pitch  |     precision: 0.724
#melody |         VFA: 0.005  ||  pitch  |        recall: 0.718
#melody |*        RPA: 0.838  ||  pitch  |            f1: 0.719
#melody |         RCA: 0.839  ||  pitch  | overlap_ratio: 0.961
#melody |*         OA: 0.853

# *
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 231217-me7-06 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231217-me7-06/model_ckpt_best.pt"
#onset  |   precision: 0.931  ||  offset |     precision: 0.924
#onset  |      recall: 0.951  ||  offset |        recall: 0.943
#onset  |*         f1: 0.937  ||  offset |*           f1: 0.930
#melody |          VR: 0.997  ||  pitch  |     precision: 0.720
#melody |         VFA: 0.005  ||  pitch  |        recall: 0.734
#melody |*        RPA: 0.846  ||  pitch  |            f1: 0.725
#melody |         RCA: 0.846  ||  pitch  | overlap_ratio: 0.961
#melody |*         OA: 0.860

CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 231219-me8-01 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231219-me8-01/model_ckpt_best.pt"
#onset  |   precision: 0.074  ||  offset |     precision: 0.247
#onset  |      recall: 0.006  ||  offset |        recall: 0.035
#onset  |*         f1: 0.010  ||  offset |*           f1: 0.058
#melody |          VR: 1.000  ||  pitch  |     precision: 0.000
#melody |         VFA: 0.998  ||  pitch  |        recall: 0.000
#melody |*        RPA: 0.324  ||  pitch  |            f1: 0.000
#melody |         RCA: 0.329  ||  pitch  | overlap_ratio: 0.001
#melody |*         OA: 0.294
# NOTE: 这是 regulate_boundary 出了bug！！！下面才是对的！！！
#onset  |   precision: 0.965  ||  offset |     precision: 0.961
#onset  |      recall: 0.918  ||  offset |        recall: 0.914
#onset  |*         f1: 0.937  ||  offset |*           f1: 0.933
#melody |          VR: 0.998  ||  pitch  |     precision: 0.753
#melody |         VFA: 0.007  ||  pitch  |        recall: 0.721
#melody |*        RPA: 0.844  ||  pitch  |            f1: 0.735
#melody |         RCA: 0.844  ||  pitch  | overlap_ratio: 0.968
#melody |*         OA: 0.858

CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 231219-me8-02 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231219-me8-02/model_ckpt_best.pt"
#onset  |   precision: 0.074  ||  offset |     precision: 0.247
#onset  |      recall: 0.005  ||  offset |        recall: 0.035
#onset  |*         f1: 0.010  ||  offset |*           f1: 0.057
#melody |          VR: 1.000  ||  pitch  |     precision: 0.000
#melody |         VFA: 0.999  ||  pitch  |        recall: 0.000
#melody |*        RPA: 0.316  ||  pitch  |            f1: 0.000
#melody |         RCA: 0.321  ||  pitch  | overlap_ratio: 0.000
#melody |*         OA: 0.286
# NOTE: 这是 regulate_boundary 出了bug！！！下面才是对的！！！
#onset  |   precision: 0.970  ||  offset |     precision: 0.967
 #onset  |      recall: 0.891  ||  offset |        recall: 0.888
 #onset  |*         f1: 0.924  ||  offset |*           f1: 0.921
 #melody |          VR: 0.997  ||  pitch  |     precision: 0.728
 #melody |         VFA: 0.006  ||  pitch  |        recall: 0.679
 #melody |*        RPA: 0.826  ||  pitch  |            f1: 0.700
 #melody |         RCA: 0.826  ||  pitch  | overlap_ratio: 0.967
 #melody |*         OA: 0.842

CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 231219-me8-04 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231219-me8-04/model_ckpt_best.pt"
#onset  |   precision: 0.949  ||  offset |     precision: 0.945
#onset  |      recall: 0.947  ||  offset |        recall: 0.943
#onset  |*         f1: 0.945  ||  offset |*           f1: 0.941
#melody |          VR: 0.998  ||  pitch  |     precision: 0.768
#melody |         VFA: 0.010  ||  pitch  |        recall: 0.767
#melody |*        RPA: 0.862  ||  pitch  |            f1: 0.765
#melody |         RCA: 0.862  ||  pitch  | overlap_ratio: 0.964
#melody |*         OA: 0.875

CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 231224-me8-01 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231224-me8-01/model_ckpt_best.pt"
#onset  |   precision: 0.951  ||  offset |     precision: 0.947
#onset  |      recall: 0.940  ||  offset |        recall: 0.936
#onset  |*         f1: 0.942  ||  offset |*           f1: 0.938
#melody |          VR: 0.997  ||  pitch  |     precision: 0.767
#melody |         VFA: 0.007  ||  pitch  |        recall: 0.759
#melody |*        RPA: 0.864  ||  pitch  |            f1: 0.761
#melody |         RCA: 0.864  ||  pitch  | overlap_ratio: 0.965
#melody |*         OA: 0.876

CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 231224-me8-02 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231224-me8-02/model_ckpt_best.pt"
#onset  |   precision: 0.861  ||  offset |     precision: 0.852
#onset  |      recall: 0.968  ||  offset |        recall: 0.958
#onset  |*         f1: 0.907  ||  offset |*           f1: 0.898
#melody |          VR: 0.997  ||  pitch  |     precision: 0.681
#melody |         VFA: 0.012  ||  pitch  |        recall: 0.760
#melody |*        RPA: 0.846  ||  pitch  |            f1: 0.715
#melody |         RCA: 0.847  ||  pitch  | overlap_ratio: 0.954
#melody |*         OA: 0.860

CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 231224-me8-03 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231224-me8-03/model_ckpt_best.pt"
#onset  |   precision: 0.837  ||  offset |     precision: 0.829
#onset  |      recall: 0.965  ||  offset |        recall: 0.955
#onset  |*         f1: 0.891  ||  offset |*           f1: 0.882
#melody |          VR: 0.995  ||  pitch  |     precision: 0.652
#melody |         VFA: 0.011  ||  pitch  |        recall: 0.747
#melody |*        RPA: 0.837  ||  pitch  |            f1: 0.693
#melody |         RCA: 0.840  ||  pitch  | overlap_ratio: 0.953
#melody |*         OA: 0.852

CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 231224-me8-04 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231224-me8-04/model_ckpt_best.pt"
#onset  |   precision: 0.831  ||  offset |     precision: 0.826
#onset  |      recall: 0.965  ||  offset |        recall: 0.959
#onset  |*         f1: 0.887  ||  offset |*           f1: 0.881
#melody |          VR: 0.998  ||  pitch  |     precision: 0.664
#melody |         VFA: 0.014  ||  pitch  |        recall: 0.766
#melody |*        RPA: 0.852  ||  pitch  |            f1: 0.707
#melody |         RCA: 0.854  ||  pitch  | overlap_ratio: 0.960
#melody |*         OA: 0.865

# *
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 231224-me8-05 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231224-me8-05/model_ckpt_best.pt"
#onset  |   precision: 0.919  ||  offset |     precision: 0.914
#onset  |      recall: 0.966  ||  offset |        recall: 0.960
#onset  |*         f1: 0.938  ||  offset |*           f1: 0.932
#melody |          VR: 0.997  ||  pitch  |     precision: 0.749
#melody |         VFA: 0.009  ||  pitch  |        recall: 0.782
#melody |*        RPA: 0.865  ||  pitch  |            f1: 0.763
#melody |         RCA: 0.865  ||  pitch  | overlap_ratio: 0.961
#melody |*         OA: 0.878

CUDA_VISIBLE_DEVICES=1 python tasks/run.py --exp_name 231226-me5-01 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231226-me5-01/model_ckpt_best.pt"
#onset  |   precision: 0.928  ||  offset |     precision: 0.923
#onset  |      recall: 0.965  ||  offset |        recall: 0.960
#onset  |*         f1: 0.942  ||  offset |*           f1: 0.937
#melody |          VR: 0.997  ||  pitch  |     precision: 0.743
#melody |         VFA: 0.007  ||  pitch  |        recall: 0.769
#melody |*        RPA: 0.856  ||  pitch  |            f1: 0.753
#melody |         RCA: 0.856  ||  pitch  | overlap_ratio: 0.956
#melody |*         OA: 0.869

# * ?????
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 231227-me8-01 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231227-me8-01/model_ckpt_best.pt"
#onset  |   precision: 0.908  ||  offset |     precision: 0.901
#onset  |      recall: 0.973  ||  offset |        recall: 0.966
#onset  |*         f1: 0.935  ||  offset |*           f1: 0.929
#melody |          VR: 0.997  ||  pitch  |     precision: 0.742
#melody |         VFA: 0.010  ||  pitch  |        recall: 0.790
#melody |*        RPA: 0.867  ||  pitch  |            f1: 0.763
#melody |         RCA: 0.867  ||  pitch  | overlap_ratio: 0.960
#melody |*         OA: 0.879
# if infer_regulate_real_note_itv == True:
#onset  |   precision: 0.930  ||  offset |     precision: 0.923
#onset  |      recall: 0.973  ||  offset |        recall: 0.967
#onset  |*         f1: 0.948  ||  offset |*           f1: 0.941
#melody |          VR: 0.997  ||  pitch  |     precision: 0.760
#melody |         VFA: 0.007  ||  pitch  |        recall: 0.790
#melody |*        RPA: 0.867  ||  pitch  |            f1: 0.773
#melody |         RCA: 0.867  ||  pitch  | overlap_ratio: 0.960
#melody |*         OA: 0.879
# 修改了 boundary_regulate 之后：
#onset  |   precision: 0.925  ||  offset |     precision: 0.923
#onset  |      recall: 0.975  ||  offset |        recall: 0.973
#onset  |*         f1: 0.946  ||  offset |*           f1: 0.944
#melody |          VR: 0.998  ||  pitch  |     precision: 0.759
#melody |         VFA: 0.009  ||  pitch  |        recall: 0.794
#melody |*        RPA: 0.874  ||  pitch  |            f1: 0.774
#melody |         RCA: 0.874  ||  pitch  | overlap_ratio: 0.972
#melody |*         OA: 0.885

CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 231227-me8-02 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231227-me8-02/model_ckpt_best.pt"
#onset  |   precision: 0.923  ||  offset |     precision: 0.919
#onset  |      recall: 0.961  ||  offset |        recall: 0.957
#onset  |*         f1: 0.938  ||  offset |*           f1: 0.933
#melody |          VR: 0.997  ||  pitch  |     precision: 0.747
#melody |         VFA: 0.009  ||  pitch  |        recall: 0.774
#melody |*        RPA: 0.861  ||  pitch  |            f1: 0.757
#melody |         RCA: 0.861  ||  pitch  | overlap_ratio: 0.963
#melody |*         OA: 0.873

CUDA_VISIBLE_DEVICES=1 python tasks/run.py --exp_name 231228-me8-01 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231228-me8-01/model_ckpt_best.pt"
#onset  |   precision: 0.949  ||  offset |     precision: 0.944
#onset  |      recall: 0.944  ||  offset |        recall: 0.938
#onset  |*         f1: 0.943  ||  offset |*           f1: 0.937
#melody |          VR: 0.997  ||  pitch  |     precision: 0.756
#melody |         VFA: 0.008  ||  pitch  |        recall: 0.754
#melody |*        RPA: 0.855  ||  pitch  |            f1: 0.753
#melody |         RCA: 0.855  ||  pitch  | overlap_ratio: 0.964
#melody |*         OA: 0.868

CUDA_VISIBLE_DEVICES=1 python tasks/run.py --exp_name 231228-me8-02 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231228-me8-02/model_ckpt_best.pt"
#onset  |   precision: 0.959  ||  offset |     precision: 0.956
#onset  |      recall: 0.913  ||  offset |        recall: 0.910
#onset  |*         f1: 0.931  ||  offset |*           f1: 0.928
#melody |          VR: 0.997  ||  pitch  |     precision: 0.733
#melody |         VFA: 0.006  ||  pitch  |        recall: 0.705
#melody |*        RPA: 0.835  ||  pitch  |            f1: 0.716
#melody |         RCA: 0.835  ||  pitch  | overlap_ratio: 0.968
#melody |*         OA: 0.850

CUDA_VISIBLE_DEVICES=1 python tasks/run.py --exp_name 231228-me8-03 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231228-me8-03/model_ckpt_best.pt"
#onset  |   precision: 0.961  ||  offset |     precision: 0.959
#onset  |      recall: 0.904  ||  offset |        recall: 0.901
#onset  |*         f1: 0.927  ||  offset |*           f1: 0.924
#melody |          VR: 0.997  ||  pitch  |     precision: 0.727
#melody |         VFA: 0.006  ||  pitch  |        recall: 0.693
#melody |*        RPA: 0.832  ||  pitch  |            f1: 0.707
#melody |         RCA: 0.832  ||  pitch  | overlap_ratio: 0.966
#melody |*         OA: 0.847

CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 231228-me8-04 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231228-me8-04/model_ckpt_best.pt"
#onset  |   precision: 0.968  ||  offset |     precision: 0.967
#onset  |      recall: 0.858  ||  offset |        recall: 0.857
#onset  |*         f1: 0.904  ||  offset |*           f1: 0.903
#melody |          VR: 0.996  ||  pitch  |     precision: 0.684
#melody |         VFA: 0.006  ||  pitch  |        recall: 0.620
#melody |*        RPA: 0.802  ||  pitch  |            f1: 0.647
#melody |         RCA: 0.802  ||  pitch  | overlap_ratio: 0.963
#melody |*         OA: 0.820

CUDA_VISIBLE_DEVICES=1 python tasks/run.py --exp_name 231229-me8-01 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231229-me8-01/model_ckpt_best.pt,infer_regulate_real_note_itv=True"
#onset  |   precision: 0.943  ||  offset |     precision: 0.938
#onset  |      recall: 0.954  ||  offset |        recall: 0.950
#onset  |*         f1: 0.945  ||  offset |*           f1: 0.940
#melody |          VR: 0.997  ||  pitch  |     precision: 0.756
#melody |         VFA: 0.007  ||  pitch  |        recall: 0.765
#melody |*        RPA: 0.858  ||  pitch  |            f1: 0.758
#melody |         RCA: 0.858  ||  pitch  | overlap_ratio: 0.962
#melody |*         OA: 0.871

CUDA_VISIBLE_DEVICES=1 python tasks/run.py --exp_name 231229-me8-02 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231229-me8-02/model_ckpt_best.pt"
#onset  |   precision: 0.954  ||  offset |     precision: 0.951
#onset  |      recall: 0.932  ||  offset |        recall: 0.929
#onset  |*         f1: 0.939  ||  offset |*           f1: 0.936
#melody |          VR: 0.997  ||  pitch  |     precision: 0.745
#melody |         VFA: 0.008  ||  pitch  |        recall: 0.732
#melody |*        RPA: 0.845  ||  pitch  |            f1: 0.736
#melody |         RCA: 0.845  ||  pitch  | overlap_ratio: 0.967
#melody |*         OA: 0.859

CUDA_VISIBLE_DEVICES=1 python tasks/run.py --exp_name 231229-me8-03 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231229-me8-03/model_ckpt_best.pt"
#onset  |   precision: 0.963  ||  offset |     precision: 0.960
#onset  |      recall: 0.914  ||  offset |        recall: 0.911
#onset  |*         f1: 0.934  ||  offset |*           f1: 0.931
#melody |          VR: 0.997  ||  pitch  |     precision: 0.737
#melody |         VFA: 0.008  ||  pitch  |        recall: 0.707
#melody |*        RPA: 0.835  ||  pitch  |            f1: 0.719
#melody |         RCA: 0.836  ||  pitch  | overlap_ratio: 0.967
#melody |*         OA: 0.851

CUDA_VISIBLE_DEVICES=1 python tasks/run.py --exp_name 231229-me8-04 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231229-me8-04/model_ckpt_best.pt"
#onset  |   precision: 0.981  ||  offset |     precision: 0.979
#onset  |      recall: 0.862  ||  offset |        recall: 0.861
#onset  |*         f1: 0.912  ||  offset |*           f1: 0.910
#melody |          VR: 0.996  ||  pitch  |     precision: 0.699
#melody |         VFA: 0.015  ||  pitch  |        recall: 0.630
#melody |*        RPA: 0.803  ||  pitch  |            f1: 0.660
#melody |         RCA: 0.803  ||  pitch  | overlap_ratio: 0.965
#melody |*         OA: 0.820

CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 231230-me8-01 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231230-me8-01/model_ckpt_best.pt,ds_names_in_testing=m4,gen_dir_name=m4"
#onset  |   precision: 0.941  ||  offset |     precision: 0.936
#onset  |      recall: 0.958  ||  offset |        recall: 0.953
#onset  |*         f1: 0.946  ||  offset |*           f1: 0.941
#melody |          VR: 0.997  ||  pitch  |     precision: 0.765
#melody |         VFA: 0.006  ||  pitch  |        recall: 0.777
#melody |*        RPA: 0.864  ||  pitch  |            f1: 0.769
#melody |         RCA: 0.864  ||  pitch  | overlap_ratio: 0.962
#melody |*         OA: 0.877
CUDA_VISIBLE_DEVICES=1 python tasks/run.py --exp_name 231230-me8-01 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231230-me8-01/model_ckpt_best.pt,ds_names_in_testing=rms,gen_dir_name=rms"
#onset  |   precision: 0.961  ||  offset |     precision: 0.961
#onset  |      recall: 0.982  ||  offset |        recall: 0.982
#onset  |*         f1: 0.970  ||  offset |*           f1: 0.970
#melody |          VR: 0.994  ||  pitch  |     precision: 0.791
#melody |         VFA: 0.009  ||  pitch  |        recall: 0.802
#melody |*        RPA: 0.869  ||  pitch  |            f1: 0.796
#melody |         RCA: 0.869  ||  pitch  | overlap_ratio: 0.993
#melody |*         OA: 0.880

CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 231230-me8-02 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231230-me8-02/model_ckpt_best.pt,ds_names_in_testing=m4,gen_dir_name=m4"
#onset  |   precision: 0.939  ||  offset |     precision: 0.934
#onset  |      recall: 0.951  ||  offset |        recall: 0.946
#onset  |*         f1: 0.942  ||  offset |*           f1: 0.936
#melody |          VR: 0.997  ||  pitch  |     precision: 0.751
#melody |         VFA: 0.007  ||  pitch  |        recall: 0.759
#melody |*        RPA: 0.855  ||  pitch  |            f1: 0.753
#melody |         RCA: 0.855  ||  pitch  | overlap_ratio: 0.963
#melody |*         OA: 0.868
CUDA_VISIBLE_DEVICES=1 python tasks/run.py --exp_name 231230-me8-02 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231230-me8-02/model_ckpt_best.pt,ds_names_in_testing=rms,gen_dir_name=rms"
#onset  |   precision: 0.972  ||  offset |     precision: 0.972
#onset  |      recall: 0.985  ||  offset |        recall: 0.985
#onset  |*         f1: 0.977  ||  offset |*           f1: 0.978
#melody |          VR: 0.997  ||  pitch  |     precision: 0.803
#melody |         VFA: 0.001  ||  pitch  |        recall: 0.810
#melody |*        RPA: 0.866  ||  pitch  |            f1: 0.806
#melody |         RCA: 0.867  ||  pitch  | overlap_ratio: 0.994
#melody |*         OA: 0.877

CUDA_VISIBLE_DEVICES=1 python tasks/run.py --exp_name 231230-me8-03 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231230-me8-03/model_ckpt_best.pt,ds_names_in_testing=m4,gen_dir_name=m4"
#onset  |   precision: 0.954  ||  offset |     precision: 0.951
#onset  |      recall: 0.916  ||  offset |        recall: 0.913
#onset  |*         f1: 0.930  ||  offset |*           f1: 0.926
#melody |          VR: 0.997  ||  pitch  |     precision: 0.730
#melody |         VFA: 0.009  ||  pitch  |        recall: 0.708
#melody |*        RPA: 0.836  ||  pitch  |            f1: 0.716
#melody |         RCA: 0.836  ||  pitch  | overlap_ratio: 0.963
#melody |*         OA: 0.851
CUDA_VISIBLE_DEVICES=1 python tasks/run.py --exp_name 231230-me8-03 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231230-me8-03/model_ckpt_best.pt,ds_names_in_testing=rms,gen_dir_name=rms"
#onset  |   precision: 0.985  ||  offset |     precision: 0.984
#onset  |      recall: 0.983  ||  offset |        recall: 0.982
#onset  |*         f1: 0.983  ||  offset |*           f1: 0.982
#melody |          VR: 0.999  ||  pitch  |     precision: 0.813
#melody |         VFA: 0.009  ||  pitch  |        recall: 0.812
#melody |*        RPA: 0.875  ||  pitch  |            f1: 0.812
#melody |         RCA: 0.875  ||  pitch  | overlap_ratio: 0.995
#melody |*         OA: 0.885

CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 231230-me8-04 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231230-me8-04/model_ckpt_best.pt,ds_names_in_testing=m4,gen_dir_name=m4"
#onset  |   precision: 0.955  ||  offset |     precision: 0.952
#onset  |      recall: 0.898  ||  offset |        recall: 0.896
#onset  |*         f1: 0.920  ||  offset |*           f1: 0.917
#melody |          VR: 0.997  ||  pitch  |     precision: 0.704
#melody |         VFA: 0.018  ||  pitch  |        recall: 0.672
#melody |*        RPA: 0.818  ||  pitch  |            f1: 0.684
#melody |         RCA: 0.818  ||  pitch  | overlap_ratio: 0.962
#melody |*         OA: 0.833
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 231230-me8-04 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231230-me8-04/model_ckpt_best.pt,ds_names_in_testing=rms,gen_dir_name=rms"
#onset  |   precision: 0.989  ||  offset |     precision: 0.990
#onset  |      recall: 0.971  ||  offset |        recall: 0.972
#onset  |*         f1: 0.979  ||  offset |*           f1: 0.980
#melody |          VR: 1.000  ||  pitch  |     precision: 0.805
#melody |         VFA: 0.009  ||  pitch  |        recall: 0.793
#melody |*        RPA: 0.866  ||  pitch  |            f1: 0.799
#melody |         RCA: 0.866  ||  pitch  | overlap_ratio: 0.996
#melody |*         OA: 0.877

CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 231231-me8-01 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231231-me8-01/model_ckpt_best.pt,ds_names_in_testing=m4,gen_dir_name=m4"
#onset  |   precision: 0.955  ||  offset |     precision: 0.953
#onset  |      recall: 0.849  ||  offset |        recall: 0.847
#onset  |*         f1: 0.892  ||  offset |*           f1: 0.891
#melody |          VR: 0.998  ||  pitch  |     precision: 0.608
#melody |         VFA: 0.053  ||  pitch  |        recall: 0.549
#melody |*        RPA: 0.736  ||  pitch  |            f1: 0.574
#melody |         RCA: 0.745  ||  pitch  | overlap_ratio: 0.956
#melody |*         OA: 0.758
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 231231-me8-01 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231231-me8-01/model_ckpt_best.pt,ds_names_in_testing=rms,gen_dir_name=rms"
#onset  |   precision: 0.994  ||  offset |     precision: 0.994
#onset  |      recall: 0.981  ||  offset |        recall: 0.981
#onset  |*         f1: 0.987  ||  offset |*           f1: 0.987
#melody |          VR: 1.000  ||  pitch  |     precision: 0.914
#melody |         VFA: 0.009  ||  pitch  |        recall: 0.905
#melody |*        RPA: 0.945  ||  pitch  |            f1: 0.909
#melody |         RCA: 0.945  ||  pitch  | overlap_ratio: 0.995
#melody |*         OA: 0.950

CUDA_VISIBLE_DEVICES=1 python tasks/run.py --exp_name 231231-me8-02 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231231-me8-02/model_ckpt_best.pt,ds_names_in_testing=m4,gen_dir_name=m4"
#onset  |   precision: 0.977  ||  offset |     precision: 0.974
#onset  |      recall: 0.841  ||  offset |        recall: 0.839
#onset  |*         f1: 0.899  ||  offset |*           f1: 0.896
#melody |          VR: 0.997  ||  pitch  |     precision: 0.620
#melody |         VFA: 0.032  ||  pitch  |        recall: 0.547
#melody |*        RPA: 0.738  ||  pitch  |            f1: 0.578
#melody |         RCA: 0.744  ||  pitch  | overlap_ratio: 0.963
#melody |*         OA: 0.762
CUDA_VISIBLE_DEVICES=1 python tasks/run.py --exp_name 231231-me8-02 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231231-me8-02/model_ckpt_best.pt,ds_names_in_testing=rms,gen_dir_name=rms"
#onset  |   precision: 0.997  ||  offset |     precision: 0.997
#onset  |      recall: 0.971  ||  offset |        recall: 0.970
#onset  |*         f1: 0.983  ||  offset |*           f1: 0.982
#melody |          VR: 1.000  ||  pitch  |     precision: 0.895
#melody |         VFA: 0.009  ||  pitch  |        recall: 0.875
#melody |*        RPA: 0.934  ||  pitch  |            f1: 0.884
#melody |         RCA: 0.934  ||  pitch  | overlap_ratio: 0.997
#melody |*         OA: 0.939

CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 231231-me8-03 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231231-me8-03/model_ckpt_best.pt,ds_names_in_testing=m4,gen_dir_name=m4"
#onset  |   precision: 0.984  ||  offset |     precision: 0.981
#onset  |      recall: 0.832  ||  offset |        recall: 0.831
#onset  |*         f1: 0.896  ||  offset |*           f1: 0.894
#melody |          VR: 0.996  ||  pitch  |     precision: 0.603
#melody |         VFA: 0.018  ||  pitch  |        recall: 0.523
#melody |*        RPA: 0.724  ||  pitch  |            f1: 0.558
#melody |         RCA: 0.732  ||  pitch  | overlap_ratio: 0.956
#melody |*         OA: 0.750
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 231231-me8-03 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231231-me8-03/model_ckpt_best.pt,ds_names_in_testing=rms,gen_dir_name=rms"
#onset  |   precision: 0.996  ||  offset |     precision: 0.997
#onset  |      recall: 0.969  ||  offset |        recall: 0.970
#onset  |*         f1: 0.982  ||  offset |*           f1: 0.982
#melody |          VR: 1.000  ||  pitch  |     precision: 0.871
#melody |         VFA: 0.009  ||  pitch  |        recall: 0.852
#melody |*        RPA: 0.909  ||  pitch  |            f1: 0.861
#melody |         RCA: 0.909  ||  pitch  | overlap_ratio: 0.996
#melody |*         OA: 0.915

CUDA_VISIBLE_DEVICES=1 python tasks/run.py --exp_name 231231-me8-04 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231231-me8-04/model_ckpt_best.pt,ds_names_in_testing=m4,gen_dir_name=m4"
#onset  |   precision: 0.974  ||  offset |     precision: 0.972
#onset  |      recall: 0.819  ||  offset |        recall: 0.818
#onset  |*         f1: 0.883  ||  offset |*           f1: 0.882
#melody |          VR: 0.996  ||  pitch  |     precision: 0.596
#melody |         VFA: 0.021  ||  pitch  |        recall: 0.515
#melody |*        RPA: 0.722  ||  pitch  |            f1: 0.549
#melody |         RCA: 0.727  ||  pitch  | overlap_ratio: 0.965
#melody |*         OA: 0.747
CUDA_VISIBLE_DEVICES=1 python tasks/run.py --exp_name 231231-me8-04 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/231231-me8-04/model_ckpt_best.pt,ds_names_in_testing=rms,gen_dir_name=rms"
#onset  |   precision: 0.998  ||  offset |     precision: 0.997
#onset  |      recall: 0.956  ||  offset |        recall: 0.956
#onset  |*         f1: 0.976  ||  offset |*           f1: 0.975
#melody |          VR: 1.000  ||  pitch  |     precision: 0.834
#melody |         VFA: 0.009  ||  pitch  |        recall: 0.802
#melody |*        RPA: 0.884  ||  pitch  |            f1: 0.817
#melody |         RCA: 0.886  ||  pitch  | overlap_ratio: 0.998
#melody |*         OA: 0.893

CUDA_VISIBLE_DEVICES=1 python tasks/run.py --exp_name 240101-me8-01 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/240101-me8-01/model_ckpt_best.pt,gen_dir_name=w_noise,noise_in_test=True,noise_prob=0.8"
#onset  |   precision: 0.892  ||  offset |     precision: 0.889
#onset  |      recall: 0.950  ||  offset |        recall: 0.947
#onset  |*         f1: 0.915  ||  offset |*           f1: 0.912
#melody |          VR: 0.998  ||  pitch  |     precision: 0.685
#melody |         VFA: 0.237  ||  pitch  |        recall: 0.726
#melody |*        RPA: 0.831  ||  pitch  |            f1: 0.702
#melody |         RCA: 0.835  ||  pitch  | overlap_ratio: 0.965
#melody |*         OA: 0.822
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 240101-me8-01 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/240101-me8-01/model_ckpt_best.pt,gen_dir_name=wo_noise,noise_in_test=False,noise_prob=0.8"
#onset  |   precision: 0.922  ||  offset |     precision: 0.919
#onset  |      recall: 0.974  ||  offset |        recall: 0.972
#onset  |*         f1: 0.943  ||  offset |*           f1: 0.941
#melody |          VR: 0.999  ||  pitch  |     precision: 0.752
#melody |         VFA: 0.008  ||  pitch  |        recall: 0.788
#melody |*        RPA: 0.871  ||  pitch  |            f1: 0.767
#melody |         RCA: 0.871  ||  pitch  | overlap_ratio: 0.970
#melody |*         OA: 0.883

CUDA_VISIBLE_DEVICES=1 python tasks/run.py --exp_name 240101-me8-02 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/240101-me8-02/model_ckpt_best.pt,gen_dir_name=w_noise,noise_in_test=True,noise_prob=0.8"
#onset  |   precision: 0.914  ||  offset |     precision: 0.911
#onset  |      recall: 0.979  ||  offset |        recall: 0.976
#onset  |*         f1: 0.941  ||  offset |*           f1: 0.939
#melody |          VR: 0.998  ||  pitch  |     precision: 0.744
#melody |         VFA: 0.010  ||  pitch  |        recall: 0.788
#melody |*        RPA: 0.873  ||  pitch  |            f1: 0.763
#melody |         RCA: 0.873  ||  pitch  | overlap_ratio: 0.970
#melody |*         OA: 0.884
CUDA_VISIBLE_DEVICES=1 python tasks/run.py --exp_name 240101-me8-02 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/240101-me8-02/model_ckpt_best.pt,gen_dir_name=wo_noise,noise_in_test=False,noise_prob=0.8"
#onset  |   precision: 0.917  ||  offset |     precision: 0.915
#onset  |      recall: 0.979  ||  offset |        recall: 0.977
#onset  |*         f1: 0.944  ||  offset |*           f1: 0.941
#melody |          VR: 0.999  ||  pitch  |     precision: 0.750
#melody |         VFA: 0.009  ||  pitch  |        recall: 0.792
#melody |*        RPA: 0.874  ||  pitch  |            f1: 0.768
#melody |         RCA: 0.874  ||  pitch  | overlap_ratio: 0.970
#melody |*         OA: 0.885

CUDA_VISIBLE_DEVICES=1 python tasks/run.py --exp_name 240102-me8-01 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/240102-me8-01/model_ckpt_best.pt,gen_dir_name=wo_noise,noise_in_test=False,noise_prob=0.8"
#onset  |   precision: 0.938  ||  offset |     precision: 0.936
#onset  |      recall: 0.966  ||  offset |        recall: 0.964
#onset  |*         f1: 0.948  ||  offset |*           f1: 0.946
#melody |          VR: 0.998  ||  pitch  |     precision: 0.765
#melody |         VFA: 0.007  ||  pitch  |        recall: 0.785
#melody |*        RPA: 0.868  ||  pitch  |            f1: 0.773
#melody |         RCA: 0.869  ||  pitch  | overlap_ratio: 0.970
#melody |*         OA: 0.880

CUDA_VISIBLE_DEVICES=1 python tasks/run.py --exp_name 240208-me8-01 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/240208-me8-01/model_ckpt_best.pt,gen_dir_name=wo_noise,noise_in_test=False,noise_prob=0.8"
#onset  |   precision: 0.933  ||  offset |     precision: 0.930
#onset  |      recall: 0.971  ||  offset |        recall: 0.968
#onset  |*         f1: 0.948  ||  offset |*           f1: 0.945
#melody |          VR: 0.998  ||  pitch  |     precision: 0.771
#melody |         VFA: 0.008  ||  pitch  |        recall: 0.797
#melody |*        RPA: 0.876  ||  pitch  |            f1: 0.782
#melody |         RCA: 0.876  ||  pitch  | overlap_ratio: 0.969
#melody |*         OA: 0.887
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 240208-me8-01 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/240208-me8-01/model_ckpt_best.pt,gen_dir_name=wo_noise,noise_in_test=True,noise_prob=0.8"
#onset  |   precision: 0.930  ||  offset |     precision: 0.927
#onset  |      recall: 0.971  ||  offset |        recall: 0.969
#onset  |*         f1: 0.947  ||  offset |*           f1: 0.944
#melody |          VR: 0.998  ||  pitch  |     precision: 0.766
#melody |         VFA: 0.009  ||  pitch  |        recall: 0.794
#melody |*        RPA: 0.874  ||  pitch  |            f1: 0.778
#melody |         RCA: 0.874  ||  pitch  | overlap_ratio: 0.970
#melody |*         OA: 0.885

CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 240208-me8-02 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/240208-me8-02/model_ckpt_best.pt,gen_dir_name=wo_noise,noise_in_test=False,noise_prob=0.8"
#onset  |   precision: 0.937  ||  offset |     precision: 0.935
#onset  |      recall: 0.964  ||  offset |        recall: 0.961
#onset  |*         f1: 0.947  ||  offset |*           f1: 0.945
#melody |          VR: 0.999  ||  pitch  |     precision: 0.765
#melody |         VFA: 0.008  ||  pitch  |        recall: 0.783
#melody |*        RPA: 0.868  ||  pitch  |            f1: 0.772
#melody |         RCA: 0.868  ||  pitch  | overlap_ratio: 0.970
#melody |*         OA: 0.880
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 240208-me8-02 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/240208-me8-02/model_ckpt_best.pt,gen_dir_name=wo_noise,noise_in_test=True,noise_prob=0.8"
#onset  |   precision: 0.938  ||  offset |     precision: 0.935
#onset  |      recall: 0.964  ||  offset |        recall: 0.961
#onset  |*         f1: 0.948  ||  offset |*           f1: 0.945
#melody |          VR: 0.999  ||  pitch  |     precision: 0.765
#melody |         VFA: 0.010  ||  pitch  |        recall: 0.783
#melody |*        RPA: 0.869  ||  pitch  |            f1: 0.772
#melody |         RCA: 0.869  ||  pitch  | overlap_ratio: 0.970
#melody |*         OA: 0.880

CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 240209-me8-01 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/240209-me8-01/model_ckpt_best.pt,gen_dir_name=wo_noise,noise_in_test=False,noise_prob=0.8"
#onset  |   precision: 0.930  ||  offset |     precision: 0.924
#onset  |      recall: 0.940  ||  offset |        recall: 0.935
#onset  |*         f1: 0.931  ||  offset |*           f1: 0.925
#melody |          VR: 0.998  ||  pitch  |     precision: 0.721
#melody |         VFA: 0.010  ||  pitch  |        recall: 0.728
#melody |*        RPA: 0.844  ||  pitch  |            f1: 0.722
#melody |         RCA: 0.844  ||  pitch  | overlap_ratio: 0.966
#melody |*         OA: 0.858
CUDA_VISIBLE_DEVICES=1 python tasks/run.py --exp_name 240209-me8-01 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/240209-me8-01/model_ckpt_best.pt,gen_dir_name=wo_noise,noise_in_test=True,noise_prob=0.8"
#onset  |   precision: 0.912  ||  offset |     precision: 0.921
#onset  |      recall: 0.929  ||  offset |        recall: 0.939
#onset  |*         f1: 0.916  ||  offset |*           f1: 0.926
#melody |          VR: 0.998  ||  pitch  |     precision: 0.705
#melody |         VFA: 0.012  ||  pitch  |        recall: 0.716
#melody |*        RPA: 0.841  ||  pitch  |            f1: 0.708
#melody |         RCA: 0.841  ||  pitch  | overlap_ratio: 0.968
#melody |*         OA: 0.855

CUDA_VISIBLE_DEVICES=3 python tasks/run.py --exp_name 240104-me9-01 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/240104-me9-01/model_ckpt_best.pt,gen_dir_name=wo_noise,noise_in_test=False,noise_prob=0.8"
#onset  |   precision: 0.953  ||  offset |     precision: 0.951
#onset  |      recall: 0.942  ||  offset |        recall: 0.940
#onset  |*         f1: 0.944  ||  offset |*           f1: 0.942
#melody |          VR: 0.998  ||  pitch  |     precision: 0.758
#melody |         VFA: 0.007  ||  pitch  |        recall: 0.750
#melody |*        RPA: 0.855  ||  pitch  |            f1: 0.752
#melody |         RCA: 0.855  ||  pitch  | overlap_ratio: 0.970
#melody |*         OA: 0.868

CUDA_VISIBLE_DEVICES=3 python tasks/run.py --exp_name 240104-me9-02 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/240104-me9-02/model_ckpt_best.pt,gen_dir_name=wo_noise,noise_in_test=False,noise_prob=0.8"
#onset  |   precision: 0.961  ||  offset |     precision: 0.960
#onset  |      recall: 0.920  ||  offset |        recall: 0.919
#onset  |*         f1: 0.936  ||  offset |*           f1: 0.935
#melody |          VR: 0.998  ||  pitch  |     precision: 0.757
#melody |         VFA: 0.007  ||  pitch  |        recall: 0.729
#melody |*        RPA: 0.852  ||  pitch  |            f1: 0.741
#melody |         RCA: 0.852  ||  pitch  | overlap_ratio: 0.974
#melody |*         OA: 0.866

CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 240104-me9-06 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/240104-me9-06/model_ckpt_best.pt,gen_dir_name=wo_noise,noise_in_test=False,noise_prob=0.8"
#onset  |   precision: 0.929  ||  offset |     precision: 0.927
#onset  |      recall: 0.968  ||  offset |        recall: 0.965
#onset  |*         f1: 0.944  ||  offset |*           f1: 0.942
#melody |          VR: 0.999  ||  pitch  |     precision: 0.755
#melody |         VFA: 0.007  ||  pitch  |        recall: 0.782
#melody |*        RPA: 0.871  ||  pitch  |            f1: 0.766
#melody |         RCA: 0.871  ||  pitch  | overlap_ratio: 0.970
#melody |*         OA: 0.883

CUDA_VISIBLE_DEVICES=1 python tasks/run.py --exp_name 240104-me9-11 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/240104-me9-11/model_ckpt_best.pt,gen_dir_name=wo_noise,noise_in_test=False,noise_prob=0.8"
#onset  |   precision: 0.930  ||  offset |     precision: 0.928
#onset  |      recall: 0.967  ||  offset |        recall: 0.965
#onset  |*         f1: 0.945  ||  offset |*           f1: 0.943
#melody |          VR: 0.999  ||  pitch  |     precision: 0.757
#melody |         VFA: 0.007  ||  pitch  |        recall: 0.784
#melody |*        RPA: 0.872  ||  pitch  |            f1: 0.768
#melody |         RCA: 0.872  ||  pitch  | overlap_ratio: 0.971
#melody |*         OA: 0.883

CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 240106-me9-1 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/240106-me9-1/model_ckpt_best.pt,ds_names_in_testing=m4,gen_dir_name=m4"
#onset  |   precision: 0.924  ||  offset |     precision: 0.922
#onset  |      recall: 0.969  ||  offset |        recall: 0.966
#onset  |*         f1: 0.943  ||  offset |*           f1: 0.940
#melody |          VR: 0.999  ||  pitch  |     precision: 0.749
#melody |         VFA: 0.007  ||  pitch  |        recall: 0.780
#melody |*        RPA: 0.869  ||  pitch  |            f1: 0.762
#melody |         RCA: 0.869  ||  pitch  | overlap_ratio: 0.967
#melody |*         OA: 0.881
CUDA_VISIBLE_DEVICES=1 python tasks/run.py --exp_name 240106-me9-1 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/240106-me9-1/model_ckpt_best.pt,ds_names_in_testing=rms,gen_dir_name=rms"
#onset  |   precision: 0.957  ||  offset |     precision: 0.957
#onset  |      recall: 0.996  ||  offset |        recall: 0.996
#onset  |*         f1: 0.974  ||  offset |*           f1: 0.975
#melody |          VR: 0.998  ||  pitch  |     precision: 0.779
#melody |         VFA: 0.009  ||  pitch  |        recall: 0.802
#melody |*        RPA: 0.869  ||  pitch  |            f1: 0.790
#melody |         RCA: 0.870  ||  pitch  | overlap_ratio: 0.992
#melody |*         OA: 0.879

CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 240106-me9-2 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/240106-me9-2/model_ckpt_best.pt,ds_names_in_testing=m4,gen_dir_name=m4"
#onset  |   precision: 0.939  ||  offset |     precision: 0.937
#onset  |      recall: 0.948  ||  offset |        recall: 0.946
#onset  |*         f1: 0.940  ||  offset |*           f1: 0.937
#melody |          VR: 0.999  ||  pitch  |     precision: 0.748
#melody |         VFA: 0.008  ||  pitch  |        recall: 0.755
#melody |*        RPA: 0.857  ||  pitch  |            f1: 0.749
#melody |         RCA: 0.857  ||  pitch  | overlap_ratio: 0.967
#melody |*         OA: 0.870
CUDA_VISIBLE_DEVICES=1 python tasks/run.py --exp_name 240106-me9-2 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/240106-me9-2/model_ckpt_best.pt,ds_names_in_testing=rms,gen_dir_name=rms"
#onset  |   precision: 0.976  ||  offset |     precision: 0.978
#onset  |      recall: 0.992  ||  offset |        recall: 0.993
#onset  |*         f1: 0.984  ||  offset |*           f1: 0.985
#melody |          VR: 1.000  ||  pitch  |     precision: 0.812
#melody |         VFA: 0.009  ||  pitch  |        recall: 0.822
#melody |*        RPA: 0.873  ||  pitch  |            f1: 0.817
#melody |         RCA: 0.874  ||  pitch  | overlap_ratio: 0.993
#melody |*         OA: 0.883
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 240106-me9-2 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/240106-me9-2/model_ckpt_best.pt"
#onset  |   precision: 0.942  ||  offset |     precision: 0.939
#onset  |      recall: 0.951  ||  offset |        recall: 0.949
#onset  |*         f1: 0.943  ||  offset |*           f1: 0.941
#melody |          VR: 0.999  ||  pitch  |     precision: 0.752
#melody |         VFA: 0.008  ||  pitch  |        recall: 0.760
#melody |*        RPA: 0.858  ||  pitch  |            f1: 0.754
#melody |         RCA: 0.858  ||  pitch  | overlap_ratio: 0.969
#melody |*         OA: 0.871

CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 240106-me9-3 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/240106-me9-3/model_ckpt_best.pt,ds_names_in_testing=m4,gen_dir_name=m4"
#onset  |   precision: 0.951  ||  offset |     precision: 0.949
#onset  |      recall: 0.923  ||  offset |        recall: 0.921
#onset  |*         f1: 0.932  ||  offset |*           f1: 0.930
#melody |          VR: 0.998  ||  pitch  |     precision: 0.731
#melody |         VFA: 0.008  ||  pitch  |        recall: 0.714
#melody |*        RPA: 0.840  ||  pitch  |            f1: 0.719
#melody |         RCA: 0.840  ||  pitch  | overlap_ratio: 0.966
#melody |*         OA: 0.854
CUDA_VISIBLE_DEVICES=1 python tasks/run.py --exp_name 240106-me9-3 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/240106-me9-3/model_ckpt_best.pt,ds_names_in_testing=rms,gen_dir_name=rms"
#onset  |   precision: 0.984  ||  offset |     precision: 0.983
#onset  |      recall: 0.981  ||  offset |        recall: 0.981
#onset  |*         f1: 0.982  ||  offset |*           f1: 0.981
#melody |          VR: 1.000  ||  pitch  |     precision: 0.797
#melody |         VFA: 0.009  ||  pitch  |        recall: 0.796
#melody |*        RPA: 0.853  ||  pitch  |            f1: 0.796
#melody |         RCA: 0.854  ||  pitch  | overlap_ratio: 0.995
#melody |*         OA: 0.865
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 240106-me9-3 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/240106-me9-3/model_ckpt_best.pt"
#onset  |   precision: 0.953  ||  offset |     precision: 0.952
#onset  |      recall: 0.927  ||  offset |        recall: 0.926
#onset  |*         f1: 0.935  ||  offset |*           f1: 0.934
#melody |          VR: 0.998  ||  pitch  |     precision: 0.736
#melody |         VFA: 0.008  ||  pitch  |        recall: 0.720
#melody |*        RPA: 0.841  ||  pitch  |            f1: 0.726
#melody |         RCA: 0.841  ||  pitch  | overlap_ratio: 0.968
#melody |*         OA: 0.855

CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 240106-me9-4 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/240106-me9-4/model_ckpt_best.pt,ds_names_in_testing=m4,gen_dir_name=m4"
#onset  |   precision: 0.963  ||  offset |     precision: 0.962
#onset  |      recall: 0.893  ||  offset |        recall: 0.892
#onset  |*         f1: 0.922  ||  offset |*           f1: 0.920
#melody |          VR: 0.997  ||  pitch  |     precision: 0.679
#melody |         VFA: 0.007  ||  pitch  |        recall: 0.638
#melody |*        RPA: 0.772  ||  pitch  |            f1: 0.655
#melody |         RCA: 0.773  ||  pitch  | overlap_ratio: 0.968
#melody |*         OA: 0.793
CUDA_VISIBLE_DEVICES=1 python tasks/run.py --exp_name 240106-me9-4 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/240106-me9-4/model_ckpt_best.pt,ds_names_in_testing=rms,gen_dir_name=rms"
#onset  |   precision: 0.994  ||  offset |     precision: 0.994
#onset  |      recall: 0.975  ||  offset |        recall: 0.975
#onset  |*         f1: 0.984  ||  offset |*           f1: 0.984
#melody |          VR: 1.000  ||  pitch  |     precision: 0.792
#melody |         VFA: 0.009  ||  pitch  |        recall: 0.779
#melody |*        RPA: 0.838  ||  pitch  |            f1: 0.785
#melody |         RCA: 0.841  ||  pitch  | overlap_ratio: 0.996
#melody |*         OA: 0.851
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name 240106-me9-4 --infer --hparams "load_ckpt=/mnt/sdb/liruiqi/SingingDictation/checkpoints/240106-me9-4/model_ckpt_best.pt"
#onset  |   precision: 0.966  ||  offset |     precision: 0.964
#onset  |      recall: 0.899  ||  offset |        recall: 0.898
#onset  |*         f1: 0.926  ||  offset |*           f1: 0.925
#melody |          VR: 0.997  ||  pitch  |     precision: 0.687
#melody |         VFA: 0.007  ||  pitch  |        recall: 0.649
#melody |*        RPA: 0.777  ||  pitch  |            f1: 0.665
#melody |         RCA: 0.778  ||  pitch  | overlap_ratio: 0.970
#melody |*         OA: 0.797
