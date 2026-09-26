# Threshold policies on checkpoint `ckpt_subset05_r150`

OOF entities 111329, link rows 518180, global decoy ratio r = 2.019 (density mode `global`), objective macro F0.5 on the calibrated scale, grid step 0.01.

## Configurations (nested = fit on K-1 folds, scored on the held-out fold; the honest comparison)

| config | classes | nested macro F0.5 | gain vs global | folds better | fold std | in-sample macro F0.5 | precision | recall | singleton acc | links |
|---|---|---|---|---|---|---|---|---|---|---|
| global | 0 | 0.99026 | +0.00000 | 0/4 | 0.00020 | 0.99031 | 0.99818 | 0.97644 | 0.9941 | 376635 |
| country | 2 | 0.99029 | +0.00003 | 3/4 | 0.00023 | 0.99034 | 0.99859 | 0.97527 | 0.9956 | 376029 |
| country|src | 4 | 0.99026 | +0.00001 | 2/4 | 0.00017 | 0.99037 | 0.99858 | 0.97536 | 0.9956 | 376070 |
| noaddr | 2 | 0.99024 | -0.00002 | 1/4 | 0.00019 | 0.99034 | 0.99851 | 0.97554 | 0.9951 | 376163 |
| noaddr_f03 | 2 | 0.99023 | -0.00003 | 1/4 | 0.00017 | 0.99033 | 0.99856 | 0.97533 | 0.9951 | 376061 |
| country_pfloor | 2 | 0.98971 | -0.00054 | 0/4 | 0.00036 | 0.98974 | 0.99900 | 0.97298 | 0.9968 | 374992 |
| country|src|noaddr | 8 | 0.99019 | -0.00006 | 0/4 | 0.00023 | 0.99040 | 0.99857 | 0.97538 | 0.9953 | 376080 |

**Selected: `global`** (auto: best nested macro F0.5 with gain >= 0.0001 over global, else global)

## country: per-class thresholds and metrics (OOF, in-sample)

| class | entities | link rows | decoy ratio | thr global | thr class | F0.5 global | F0.5 class | delta | P global | P class | R global | R class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| country=India | 44671 | 207618 | 2.019 | 0.72 | 0.76 | 0.98871 | 0.98872 | +0.00001 | 0.99770 | 0.99797 | 0.97325 | 0.97258 |
| country=US | 66658 | 310562 | 2.019 | 0.72 | 0.83 | 0.99139 | 0.99143 | +0.00004 | 0.99850 | 0.99900 | 0.97902 | 0.97752 |

Per-fold refits (thresholds chosen without that fold): fold 0: country=India=0.71, country=US=0.74; fold 1: country=India=0.72, country=US=0.83; fold 2: country=India=0.76, country=US=0.83; fold 3: country=India=0.76, country=US=0.83

## country|src: per-class thresholds and metrics (OOF, in-sample)

| class | entities | link rows | decoy ratio | thr global | thr class | F0.5 global | F0.5 class | delta | P global | P class | R global | R class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| country=India|src=2 | 41442 | 101310 | 2.019 | 0.72 | 0.73 | 0.98958 | 0.98966 | +0.00008 | 0.99793 | 0.99800 | 0.97731 | 0.97722 |
| country=India|src=3 | 41626 | 106308 | 2.019 | 0.72 | 0.76 | 0.98965 | 0.98975 | +0.00011 | 0.99749 | 0.99788 | 0.96946 | 0.96871 |
| country=US|src=2 | 61726 | 151682 | 2.019 | 0.72 | 0.83 | 0.99197 | 0.99201 | +0.00004 | 0.99856 | 0.99907 | 0.97829 | 0.97670 |
| country=US|src=3 | 62248 | 158880 | 2.019 | 0.72 | 0.83 | 0.99191 | 0.99196 | +0.00005 | 0.99843 | 0.99894 | 0.97970 | 0.97828 |

Per-fold refits (thresholds chosen without that fold): fold 0: country=India|src=2=0.70, country=India|src=3=0.76, country=US|src=2=0.74, country=US|src=3=0.82; fold 1: country=India|src=2=0.72, country=India|src=3=0.76, country=US|src=2=0.83, country=US|src=3=0.83; fold 2: country=India|src=2=0.84, country=India|src=3=0.76, country=US|src=2=0.72, country=US|src=3=0.76; fold 3: country=India|src=2=0.73, country=India|src=3=0.76, country=US|src=2=0.85, country=US|src=3=0.74

## noaddr: per-class thresholds and metrics (OOF, in-sample)

| class | entities | link rows | decoy ratio | thr global | thr class | F0.5 global | F0.5 class | delta | P global | P class | R global | R class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| noaddr=0 | 109874 | 500972 | 2.019 | 0.72 | 0.76 | 0.99050 | 0.99051 | +0.00002 | 0.99852 | 0.99867 | 0.99336 | 0.99296 |
| noaddr=1 | 14513 | 17208 | 2.019 | 0.72 | 0.83 | 0.98344 | 0.98363 | +0.00020 | 0.98618 | 0.99295 | 0.61337 | 0.60158 |

Per-fold refits (thresholds chosen without that fold): fold 0: noaddr=0=0.71, noaddr=1=0.82; fold 1: noaddr=0=0.79, noaddr=1=0.83; fold 2: noaddr=0=0.71, noaddr=1=0.87; fold 3: noaddr=0=0.76, noaddr=1=0.85

## noaddr_f03: per-class thresholds and metrics (OOF, in-sample)

| class | entities | link rows | decoy ratio | thr global | thr class | F0.5 global | F0.5 class | delta | P global | P class | R global | R class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| noaddr=0 | 109874 | 500972 | 2.019 | 0.72 | 0.76 | 0.99050 | 0.99052 | +0.00002 | 0.99852 | 0.99867 | 0.99336 | 0.99296 |
| noaddr=1 | 14513 | 17208 | 2.019 | 0.72 | 0.87 | 0.98344 | 0.98360 | +0.00016 | 0.98618 | 0.99466 | 0.61337 | 0.59660 |

Per-fold refits (thresholds chosen without that fold): fold 0: noaddr=0=0.71, noaddr=1=0.89; fold 1: noaddr=0=0.79, noaddr=1=0.89; fold 2: noaddr=0=0.71, noaddr=1=0.89; fold 3: noaddr=0=0.76, noaddr=1=0.87

## country_pfloor: per-class thresholds and metrics (OOF, in-sample)

| class | entities | link rows | decoy ratio | thr global | thr class | F0.5 global | F0.5 class | delta | P global | P class | R global | R class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| country=India | 44671 | 207618 | 2.019 | 0.72 | 0.93 | 0.98871 | 0.98723 | -0.00149 | 0.99770 | 0.99900 | 0.97325 | 0.96687 |
| country=US | 66658 | 310562 | 2.019 | 0.72 | 0.83 | 0.99139 | 0.99143 | +0.00004 | 0.99850 | 0.99900 | 0.97902 | 0.97752 |

Per-fold refits (thresholds chosen without that fold): fold 0: country=India=0.93, country=US=0.83; fold 1: country=India=0.93, country=US=0.81; fold 2: country=India=0.93, country=US=0.85; fold 3: country=India=0.94, country=US=0.84

## country|src|noaddr: per-class thresholds and metrics (OOF, in-sample)

| class | entities | link rows | decoy ratio | thr global | thr class | F0.5 global | F0.5 class | delta | P global | P class | R global | R class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| country=India|src=2|noaddr=0 | 41086 | 98408 | 2.019 | 0.72 | 0.71 | 0.98968 | 0.98983 | +0.00015 | 0.99827 | 0.99824 | 0.99087 | 0.99094 |
| country=India|src=2|noaddr=1 | 2709 | 2902 | 2.019 | 0.72 | 0.89 | 0.98322 | 0.98314 | -0.00008 | 0.98479 | 0.99376 | 0.63636 | 0.61495 |
| country=India|src=3|noaddr=0 | 41288 | 103067 | 2.019 | 0.72 | 0.76 | 0.98988 | 0.98996 | +0.00008 | 0.99777 | 0.99808 | 0.98603 | 0.98539 |
| country=India|src=3|noaddr=1 | 2910 | 3241 | 2.019 | 0.72 | 0.87 | 0.97978 | 0.98091 | +0.00113 | 0.98592 | 0.99661 | 0.57107 | 0.55287 |
| country=US|src=2|noaddr=0 | 61050 | 146204 | 2.019 | 0.72 | 0.80 | 0.99210 | 0.99217 | +0.00007 | 0.99893 | 0.99915 | 0.99643 | 0.99581 |
| country=US|src=2|noaddr=1 | 4931 | 5478 | 2.019 | 0.72 | 0.73 | 0.98403 | 0.98426 | +0.00022 | 0.98698 | 0.98784 | 0.62019 | 0.61945 |
| country=US|src=3|noaddr=0 | 61684 | 153293 | 2.019 | 0.72 | 0.82 | 0.99207 | 0.99218 | +0.00011 | 0.99880 | 0.99910 | 0.99701 | 0.99612 |
| country=US|src=3|noaddr=1 | 5002 | 5587 | 2.019 | 0.72 | 0.85 | 0.98365 | 0.98370 | +0.00005 | 0.98628 | 0.99458 | 0.61936 | 0.60598 |

Per-fold refits (thresholds chosen without that fold): fold 0: country=India|src=2|noaddr=0=0.70, country=India|src=2|noaddr=1=0.91, country=India|src=3|noaddr=0=0.70, country=India|src=3|noaddr=1=0.87, country=US|src=2|noaddr=0=0.74, country=US|src=2|noaddr=1=0.73, country=US|src=3|noaddr=0=0.82, country=US|src=3|noaddr=1=0.72; fold 1: country=India|src=2|noaddr=0=0.72, country=India|src=2|noaddr=1=0.72, country=India|src=3|noaddr=0=0.76, country=India|src=3|noaddr=1=0.87, country=US|src=2|noaddr=0=0.80, country=US|src=2|noaddr=1=0.73, country=US|src=3|noaddr=0=0.82, country=US|src=3|noaddr=1=0.85; fold 2: country=India|src=2|noaddr=0=0.70, country=India|src=2|noaddr=1=0.89, country=India|src=3|noaddr=0=0.72, country=India|src=3|noaddr=1=0.87, country=US|src=2|noaddr=0=0.72, country=US|src=2|noaddr=1=0.83, country=US|src=3|noaddr=0=0.76, country=US|src=3|noaddr=1=0.85; fold 3: country=India|src=2|noaddr=0=0.71, country=India|src=2|noaddr=1=0.80, country=India|src=3|noaddr=0=0.76, country=India|src=3|noaddr=1=0.75, country=US|src=2|noaddr=0=0.80, country=US|src=2|noaddr=1=0.85, country=US|src=3|noaddr=0=0.74, country=US|src=3|noaddr=1=0.87

