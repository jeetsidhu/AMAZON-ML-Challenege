# Threshold policies on checkpoint `E2_c5`

OOF entities 111329, link rows 518180, global decoy ratio r = 2.019 (density mode `global`), objective macro F0.5 on the calibrated scale, grid step 0.01.

## Configurations (nested = fit on K-1 folds, scored on the held-out fold; the honest comparison)

| config | classes | nested macro F0.5 | gain vs global | folds better | fold std | in-sample macro F0.5 | precision | recall | singleton acc | links |
|---|---|---|---|---|---|---|---|---|---|---|
| global | 0 | 0.99127 | +0.00000 | 0/4 | 0.00023 | 0.99131 | 0.99866 | 0.97761 | 0.9957 | 376903 |
| country | 2 | 0.99122 | -0.00005 | 1/4 | 0.00018 | 0.99133 | 0.99869 | 0.97755 | 0.9959 | 376872 |
| src | 2 | 0.99128 | +0.00001 | 1/4 | 0.00025 | 0.99132 | 0.99863 | 0.97772 | 0.9957 | 376958 |
| country|src | 4 | 0.99126 | -0.00000 | 2/4 | 0.00022 | 0.99134 | 0.99876 | 0.97737 | 0.9960 | 376774 |
| indic | 2 | 0.99126 | -0.00001 | 3/4 | 0.00030 | 0.99132 | 0.99867 | 0.97760 | 0.9957 | 376895 |
| noaddr | 2 | 0.99125 | -0.00001 | 1/4 | 0.00024 | 0.99131 | 0.99868 | 0.97758 | 0.9957 | 376885 |
| country|src|indic | 6 | 0.99123 | -0.00004 | 2/4 | 0.00026 | 0.99135 | 0.99872 | 0.97748 | 0.9959 | 376835 |
| country|src|noaddr | 8 | 0.99128 | +0.00001 | 2/4 | 0.00022 | 0.99137 | 0.99863 | 0.97781 | 0.9954 | 376992 |

**Selected: `global`** (auto: best nested macro F0.5 with gain >= 0.0001 over global, else global)

## country: per-class thresholds and metrics (OOF, in-sample)

| class | entities | link rows | decoy ratio | thr global | thr class | F0.5 global | F0.5 class | delta | P global | P class | R global | R class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| country=India | 44671 | 207618 | 2.019 | 0.76 | 0.77 | 0.99037 | 0.99040 | +0.00004 | 0.99832 | 0.99838 | 0.97565 | 0.97551 |
| country=US | 66658 | 310562 | 2.019 | 0.76 | 0.76 | 0.99194 | 0.99194 | +0.00000 | 0.99889 | 0.99889 | 0.97936 | 0.97936 |

Per-fold refits (thresholds chosen without that fold): fold 0: country=India=0.77, country=US=0.76; fold 1: country=India=0.78, country=US=0.74; fold 2: country=India=0.74, country=US=0.75; fold 3: country=India=0.77, country=US=0.83

## src: per-class thresholds and metrics (OOF, in-sample)

| class | entities | link rows | decoy ratio | thr global | thr class | F0.5 global | F0.5 class | delta | P global | P class | R global | R class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| src=2 | 104020 | 252992 | 2.019 | 0.76 | 0.74 | 0.99178 | 0.99179 | +0.00001 | 0.99871 | 0.99864 | 0.97882 | 0.97905 |
| src=3 | 104722 | 265188 | 2.019 | 0.76 | 0.76 | 0.99180 | 0.99181 | +0.00000 | 0.99862 | 0.99862 | 0.97699 | 0.97699 |

Per-fold refits (thresholds chosen without that fold): fold 0: src=2=0.74, src=3=0.76; fold 1: src=2=0.75, src=3=0.76; fold 2: src=2=0.74, src=3=0.74; fold 3: src=2=0.74, src=3=0.77

## country|src: per-class thresholds and metrics (OOF, in-sample)

| class | entities | link rows | decoy ratio | thr global | thr class | F0.5 global | F0.5 class | delta | P global | P class | R global | R class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| country=India|src=2 | 41884 | 101310 | 2.019 | 0.76 | 0.78 | 0.99078 | 0.99082 | +0.00004 | 0.99833 | 0.99841 | 0.97896 | 0.97873 |
| country=India|src=3 | 41956 | 106308 | 2.019 | 0.76 | 0.77 | 0.99102 | 0.99102 | +0.00000 | 0.99830 | 0.99838 | 0.97256 | 0.97238 |
| country=US|src=2 | 62136 | 151682 | 2.019 | 0.76 | 0.76 | 0.99246 | 0.99248 | +0.00002 | 0.99896 | 0.99896 | 0.97872 | 0.97872 |
| country=US|src=3 | 62766 | 158880 | 2.019 | 0.76 | 0.81 | 0.99232 | 0.99235 | +0.00003 | 0.99883 | 0.99906 | 0.97996 | 0.97946 |

Per-fold refits (thresholds chosen without that fold): fold 0: country=India|src=2=0.78, country=India|src=3=0.77, country=US|src=2=0.76, country=US|src=3=0.81; fold 1: country=India|src=2=0.78, country=India|src=3=0.78, country=US|src=2=0.74, country=US|src=3=0.76; fold 2: country=India|src=2=0.74, country=India|src=3=0.70, country=US|src=2=0.76, country=US|src=3=0.81; fold 3: country=India|src=2=0.77, country=India|src=3=0.76, country=US|src=2=0.76, country=US|src=3=0.81

## indic: per-class thresholds and metrics (OOF, in-sample)

| class | entities | link rows | decoy ratio | thr global | thr class | F0.5 global | F0.5 class | delta | P global | P class | R global | R class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| indic=0 | 109235 | 479636 | 2.019 | 0.76 | 0.76 | 0.99149 | 0.99150 | +0.00001 | 0.99867 | 0.99867 | 0.97726 | 0.97726 |
| indic=1 | 16463 | 38544 | 2.019 | 0.76 | 0.77 | 0.98872 | 0.98877 | +0.00005 | 0.99857 | 0.99872 | 0.98557 | 0.98543 |

Per-fold refits (thresholds chosen without that fold): fold 0: indic=0=0.76, indic=1=0.48; fold 1: indic=0=0.76, indic=1=0.77; fold 2: indic=0=0.74, indic=1=0.77; fold 3: indic=0=0.76, indic=1=0.77

## noaddr: per-class thresholds and metrics (OOF, in-sample)

| class | entities | link rows | decoy ratio | thr global | thr class | F0.5 global | F0.5 class | delta | P global | P class | R global | R class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| noaddr=0 | 110483 | 500972 | 2.019 | 0.76 | 0.76 | 0.99141 | 0.99141 | +0.00000 | 0.99888 | 0.99888 | 0.99450 | 0.99450 |
| noaddr=1 | 14749 | 17208 | 2.019 | 0.76 | 0.77 | 0.98448 | 0.98451 | +0.00002 | 0.99102 | 0.99158 | 0.61515 | 0.61444 |

Per-fold refits (thresholds chosen without that fold): fold 0: noaddr=0=0.76, noaddr=1=0.85; fold 1: noaddr=0=0.74, noaddr=1=0.77; fold 2: noaddr=0=0.74, noaddr=1=0.84; fold 3: noaddr=0=0.77, noaddr=1=0.77

## country|src|indic: per-class thresholds and metrics (OOF, in-sample)

| class | entities | link rows | decoy ratio | thr global | thr class | F0.5 global | F0.5 class | delta | P global | P class | R global | R class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| country=India|src=2|indic=0 | 35902 | 76990 | 2.019 | 0.76 | 0.82 | 0.99110 | 0.99114 | +0.00004 | 0.99832 | 0.99865 | 0.97699 | 0.97602 |
| country=India|src=2|indic=1 | 14838 | 24320 | 2.019 | 0.76 | 0.77 | 0.98919 | 0.98925 | +0.00006 | 0.99835 | 0.99852 | 0.98526 | 0.98509 |
| country=India|src=3|indic=0 | 39770 | 92084 | 2.019 | 0.76 | 0.69 | 0.99104 | 0.99104 | -0.00001 | 0.99820 | 0.99777 | 0.97047 | 0.97150 |
| country=India|src=3|indic=1 | 9950 | 14224 | 2.019 | 0.76 | 0.79 | 0.99054 | 0.99057 | +0.00003 | 0.99895 | 0.99914 | 0.98610 | 0.98600 |
| country=US|src=2|indic=0 | 62136 | 151682 | 2.019 | 0.76 | 0.76 | 0.99246 | 0.99248 | +0.00002 | 0.99896 | 0.99896 | 0.97872 | 0.97872 |
| country=US|src=3|indic=0 | 62766 | 158880 | 2.019 | 0.76 | 0.81 | 0.99232 | 0.99235 | +0.00003 | 0.99883 | 0.99906 | 0.97996 | 0.97946 |

Per-fold refits (thresholds chosen without that fold): fold 0: country=India|src=2|indic=0=0.83, country=India|src=2|indic=1=0.51, country=India|src=3|indic=0=0.75, country=India|src=3|indic=1=0.79, country=US|src=2|indic=0=0.76, country=US|src=3|indic=0=0.81; fold 1: country=India|src=2|indic=0=0.79, country=India|src=2|indic=1=0.77, country=India|src=3|indic=0=0.78, country=India|src=3|indic=1=0.79, country=US|src=2|indic=0=0.74, country=US|src=3|indic=0=0.76; fold 2: country=India|src=2|indic=0=0.83, country=India|src=2|indic=1=0.77, country=India|src=3|indic=0=0.70, country=India|src=3|indic=1=0.77, country=US|src=2|indic=0=0.76, country=US|src=3|indic=0=0.81; fold 3: country=India|src=2|indic=0=0.74, country=India|src=2|indic=1=0.77, country=India|src=3|indic=0=0.69, country=India|src=3|indic=1=0.79, country=US|src=2|indic=0=0.76, country=US|src=3|indic=0=0.81

## country|src|noaddr: per-class thresholds and metrics (OOF, in-sample)

| class | entities | link rows | decoy ratio | thr global | thr class | F0.5 global | F0.5 class | delta | P global | P class | R global | R class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| country=India|src=2|noaddr=0 | 41549 | 98408 | 2.019 | 0.76 | 0.78 | 0.99088 | 0.99098 | +0.00010 | 0.99853 | 0.99861 | 0.99252 | 0.99234 |
| country=India|src=2|noaddr=1 | 2735 | 2902 | 2.019 | 0.76 | 0.74 | 0.98393 | 0.98434 | +0.00040 | 0.99074 | 0.99077 | 0.63812 | 0.64022 |
| country=India|src=3|noaddr=0 | 41624 | 103067 | 2.019 | 0.76 | 0.69 | 0.99118 | 0.99126 | +0.00008 | 0.99849 | 0.99829 | 0.98909 | 0.98980 |
| country=India|src=3|noaddr=1 | 2960 | 3241 | 2.019 | 0.76 | 0.85 | 0.98293 | 0.98299 | +0.00006 | 0.99081 | 0.99559 | 0.57484 | 0.56699 |
| country=US|src=2|noaddr=0 | 61430 | 146204 | 2.019 | 0.76 | 0.74 | 0.99260 | 0.99261 | +0.00001 | 0.99918 | 0.99915 | 0.99677 | 0.99693 |
| country=US|src=2|noaddr=1 | 4990 | 5478 | 2.019 | 0.76 | 0.70 | 0.98478 | 0.98490 | +0.00012 | 0.99200 | 0.98717 | 0.62242 | 0.62930 |
| country=US|src=3|noaddr=0 | 62170 | 153293 | 2.019 | 0.76 | 0.74 | 0.99251 | 0.99254 | +0.00004 | 0.99908 | 0.99906 | 0.99727 | 0.99743 |
| country=US|src=3|noaddr=1 | 5055 | 5587 | 2.019 | 0.76 | 0.81 | 0.98328 | 0.98353 | +0.00026 | 0.99033 | 0.99350 | 0.61955 | 0.61643 |

Per-fold refits (thresholds chosen without that fold): fold 0: country=India|src=2|noaddr=0=0.78, country=India|src=2|noaddr=1=0.74, country=India|src=3|noaddr=0=0.69, country=India|src=3|noaddr=1=0.76, country=US|src=2|noaddr=0=0.75, country=US|src=2|noaddr=1=0.69, country=US|src=3|noaddr=0=0.81, country=US|src=3|noaddr=1=0.82; fold 1: country=India|src=2|noaddr=0=0.78, country=India|src=2|noaddr=1=0.91, country=India|src=3|noaddr=0=0.78, country=India|src=3|noaddr=1=0.85, country=US|src=2|noaddr=0=0.74, country=US|src=2|noaddr=1=0.70, country=US|src=3|noaddr=0=0.74, country=US|src=3|noaddr=1=0.82; fold 2: country=India|src=2|noaddr=0=0.78, country=India|src=2|noaddr=1=0.74, country=India|src=3|noaddr=0=0.69, country=India|src=3|noaddr=1=0.80, country=US|src=2|noaddr=0=0.74, country=US|src=2|noaddr=1=0.76, country=US|src=3|noaddr=0=0.74, country=US|src=3|noaddr=1=0.81; fold 3: country=India|src=2|noaddr=0=0.77, country=India|src=2|noaddr=1=0.74, country=India|src=3|noaddr=0=0.69, country=India|src=3|noaddr=1=0.76, country=US|src=2|noaddr=0=0.74, country=US|src=2|noaddr=1=0.70, country=US|src=3|noaddr=0=0.81, country=US|src=3|noaddr=1=0.77

