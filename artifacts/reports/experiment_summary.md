| experiment_name | backbone | loss | class_weights | weighted_sampler | label_smoothing | best_val_macro_f1 | test_macro_f1 | balanced_accuracy | accuracy | mel_recall | bcc_recall | akiec_recall | needs_review_rate | checkpoint_epoch | top_confusion_pair | top_confusion_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| {} | convnext_tiny | cb_focal | False | False | 0.0000 | 0.8108 | 0.7743 | 0.7638 | 0.8673 | 0.5268 | 0.9216 | 0.8438 | 0.0659 | 32 | mel->nv | 42 |
| {} | convnext_tiny | cross_entropy | True | False | 0.0500 | 0.7891 | 0.7654 | 0.8054 | 0.8543 | 0.5804 | 0.9412 | 0.7188 | 0.7465 | 20 | mel->nv | 34 |
| {} | convnext_tiny | cross_entropy | False | True | 0.0500 | 0.7880 | 0.7292 | 0.7619 | 0.8174 | 0.6071 | 0.9020 | 0.6562 | 0.1148 | 9 | nv->mel | 52 |
| {} | convnext_tiny | focal | False | False | 0.0000 | 0.7836 | 0.7007 | 0.6903 | 0.8373 | 0.5625 | 0.9216 | 0.6562 | 0.1407 | 15 | mel->nv | 38 |
| {} | convnext_tiny | cross_entropy | True | False |  | 0.7687 | 0.7495 | 0.7830 | 0.8214 | 0.5982 | 0.9412 | 0.7500 | 0.0938 | 15 | nv->mel | 51 |
| {} | efficientnet_b0 | {} | True | {} |  | 0.7031 | 0.6313 | 0.6770 | 0.7814 | 0.5982 | 0.7451 | 0.5312 | 0.1457 | 8 | nv->mel | 67 |
