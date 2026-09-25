## Out-of-fold validation

pairs 1544244, positive pairs 380479, Source 1 entities 111329, stage-1 features 82, folds [0, 1, 2, 3]

| metric | value |
|---|---|
| threshold | 0.79000 |
| macro_f05 | 0.99088 |
| micro_precision | 0.99852 |
| micro_recall | 0.97562 |
| singleton_acc | 0.99493 |
| links | 376189 |
| pair-level accuracy | 0.99650 |
| pair-level positive rate | 0.24639 |
| pair-level precision | 0.99851 |
| pair-level recall | 0.98727 |
| retrieval recall@1 / @3 / @5 / @10 | 0.9792 / 0.9882 / 0.9906 / 0.9931 |

### Per fold (at the global threshold)

| fold | entities | macro F0.5 | precision | recall | singleton acc | own best thr | F0.5 @ own thr |
|---|---|---|---|---|---|---|---|
| 0 | 27606 | 0.99039 | 0.99857 | 0.97541 | 0.9943 | 0.77 | 0.99048 |
| 1 | 27588 | 0.99119 | 0.99838 | 0.97696 | 0.9949 | 0.79 | 0.99119 |
| 2 | 27872 | 0.99094 | 0.99858 | 0.97543 | 0.9949 | 0.79 | 0.99094 |
| 3 | 28263 | 0.99100 | 0.99855 | 0.97470 | 0.9956 | 0.70 | 0.99106 |

mean 0.99088, std 0.00030, min 0.99039; fold-optimal thresholds [0.7, 0.79] (std 0.0370)

### Per country

| country | entities-with-links | macro F0.5 | precision | recall |
|---|---|---|---|---|
| India | 150339 | 0.98959 | 0.99813 | 0.97209 |
| US | 225850 | 0.99175 | 0.99878 | 0.97798 |

### Calibration (nested: fit on K-1 folds, scored on the held-out fold)

| probabilities | log loss | Brier | ECE |
|---|---|---|---|
| uncalibrated | 0.00828 | 0.00238 | 0.00086 |
| platt | 0.00802 | 0.00234 | 0.00032 |
| isotonic | 0.00803 | 0.00234 | 0.00025 |

Platt: a=0.805, b=-0.065

### Cross-fold audit

cross-fold pair share 0.3679, entities with a cross-fold pair 0.8241

| entity set | macro F0.5 | precision | recall |
|---|---|---|---|
| within_fold_only | 0.99086 | 0.99884 | 0.97454 |
| with_cross_fold_pairs | 0.99089 | 0.99845 | 0.97584 |

### Threshold sweep (calibrated scale, OOF)

| thr | macro F0.5 | precision | recall | links |
|---|---|---|---|---|
| 0.10 | 0.98384 | 0.98663 | 0.98272 | 383492 |
| 0.20 | 0.98631 | 0.99008 | 0.98218 | 381948 |
| 0.30 | 0.98778 | 0.99226 | 0.98151 | 380849 |
| 0.40 | 0.98889 | 0.99412 | 0.98050 | 379748 |
| 0.50 | 0.98982 | 0.99576 | 0.97932 | 378665 |
| 0.60 | 0.99040 | 0.99697 | 0.97809 | 377726 |
| 0.70 | 0.99077 | 0.99792 | 0.97681 | 376876 |
| 0.79 | 0.99088 | 0.99852 | 0.97562 | 376189 |
| 0.80 | 0.99086 | 0.99855 | 0.97546 | 376116 |
| 0.90 | 0.99018 | 0.99904 | 0.97288 | 374940 |

## Leakage check

result: **OK**; problems: none

| check | value |
|---|---|
| name groups split across folds | 0 / 75796 |
| matched records outside their entity's fold | 0 |
| duplicated pids in OOF | 0 |
| positive pairs across folds | 0 |
| fold-only Indic tokens leaked into own-fold lexicon | 0 / 80 checked |
| label-shuffle canary OOF AUC | 0.4999 |
| strongest single features (AUC) | [['gap_second', 0.9926], ['score', 0.9797], ['cos_addr', 0.9595], ['ad_jacc', 0.959], ['ad_tset', 0.9542]] |
| bookkeeping columns among model features | none |

## Threshold transfer to the test split

decoy-density ratio r = 2.019; pair-level odds ratio for the prior shift = 0.742; chosen: density

| method | threshold | OOF macro F0.5 (train mix) | OOF macro F0.5 (test-like mix) | precision | recall |
|---|---|---|---|---|---|
| train | 0.790 | 0.99088 | 0.99042 | 0.99741 | 0.97562 |
| density | 0.790 | 0.99088 | 0.99042 | 0.99741 | 0.97562 |
| prior | 0.835 | 0.99070 | 0.99028 | 0.99772 | 0.97479 |

## Runtime profile

| stage | wall s | cpu s | cpu/wall | workers |
|---|---|---|---|---|
| folds | 2.4 | 2.7 | 1.11 | 4 |
| lexicon | 1.3 | 1.7 | 1.26 | 4 |
| prepare_train | 17.9 | 60.1 | 3.37 | 4 |
| blocking_train | 18.2 | 43.7 | 2.40 | 4 |
| features_train | 27.2 | 63.6 | 2.34 | 4 |
| prepare_test | 10.3 | 29.1 | 2.82 | 4 |
| blocking_test | 6.6 | 17.6 | 2.66 | 4 |
| features_test | 11.2 | 29.6 | 2.65 | 4 |
| train | 265.6 | 950.4 | 3.58 | 4 |
| leakage_check | 19.9 | 51.2 | 2.58 | 4 |
| predict | 11.4 | 33.8 | 2.97 | 4 |
| **total** | 391.9 | | | |
