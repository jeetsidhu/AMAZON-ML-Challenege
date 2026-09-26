# Threshold policies on checkpoint `ckpt_subset20_r150`

OOF entities 442586, link rows 2065311, global decoy ratio r = 2.022 (density mode `global`), objective macro F0.5 on the calibrated scale, grid step 0.01.

## Configurations (nested = fit on K-1 folds, scored on the held-out fold; the honest comparison)

| config | classes | nested macro F0.5 | gain vs global | folds better | fold std | in-sample macro F0.5 | precision | recall | singleton acc | links |
|---|---|---|---|---|---|---|---|---|---|---|
| global | 0 | 0.98736 | +0.00000 | 0/4 | 0.00003 | 0.98736 | 0.99772 | 0.97071 | 0.9925 | 1489069 |
| country | 2 | 0.98734 | -0.00001 | 0/4 | 0.00003 | 0.98736 | 0.99769 | 0.97078 | 0.9926 | 1489235 |
| src | 2 | 0.98732 | -0.00003 | 0/4 | 0.00001 | 0.98736 | 0.99775 | 0.97062 | 0.9926 | 1488880 |
| country|src | 4 | 0.98734 | -0.00001 | 0/4 | 0.00003 | 0.98738 | 0.99776 | 0.97056 | 0.9929 | 1488766 |
| indic | 2 | 0.98736 | +0.00000 | 2/4 | 0.00003 | 0.98736 | 0.99771 | 0.97073 | 0.9925 | 1489111 |
| noaddr | 2 | 0.98732 | -0.00004 | 0/4 | 0.00002 | 0.98736 | 0.99772 | 0.97071 | 0.9925 | 1489069 |
| country|src|indic | 6 | 0.98733 | -0.00003 | 1/4 | 0.00005 | 0.98739 | 0.99776 | 0.97057 | 0.9929 | 1488786 |
| country|src|noaddr | 8 | 0.98732 | -0.00003 | 0/4 | 0.00003 | 0.98739 | 0.99777 | 0.97058 | 0.9927 | 1488795 |

**Selected: `global`** (auto: best nested macro F0.5 with gain >= 0.0001 over global, else global)

## country: per-class thresholds and metrics (OOF, in-sample)

| class | entities | link rows | decoy ratio | thr global | thr class | F0.5 global | F0.5 class | delta | P global | P class | R global | R class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| country=India | 177325 | 827175 | 2.022 | 0.73 | 0.71 | 0.98461 | 0.98462 | +0.00001 | 0.99722 | 0.99705 | 0.96592 | 0.96629 |
| country=US | 265261 | 1238136 | 2.022 | 0.73 | 0.74 | 0.98919 | 0.98920 | +0.00001 | 0.99805 | 0.99811 | 0.97485 | 0.97473 |

Per-fold refits (thresholds chosen without that fold): fold 0: country=India=0.71, country=US=0.74; fold 1: country=India=0.71, country=US=0.74; fold 2: country=India=0.71, country=US=0.76; fold 3: country=India=0.73, country=US=0.76

## src: per-class thresholds and metrics (OOF, in-sample)

| class | entities | link rows | decoy ratio | thr global | thr class | F0.5 global | F0.5 class | delta | P global | P class | R global | R class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| src=2 | 411343 | 1007268 | 2.022 | 0.73 | 0.73 | 0.98847 | 0.98848 | +0.00001 | 0.99782 | 0.99782 | 0.97272 | 0.97272 |
| src=3 | 413790 | 1058043 | 2.022 | 0.73 | 0.74 | 0.98861 | 0.98862 | +0.00001 | 0.99763 | 0.99769 | 0.96992 | 0.96973 |

Per-fold refits (thresholds chosen without that fold): fold 0: src=2=0.74, src=3=0.73; fold 1: src=2=0.72, src=3=0.74; fold 2: src=2=0.73, src=3=0.71; fold 3: src=2=0.73, src=3=0.74

## country|src: per-class thresholds and metrics (OOF, in-sample)

| class | entities | link rows | decoy ratio | thr global | thr class | F0.5 global | F0.5 class | delta | P global | P class | R global | R class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| country=India|src=2 | 164231 | 403650 | 2.022 | 0.73 | 0.72 | 0.98611 | 0.98612 | +0.00001 | 0.99752 | 0.99744 | 0.97145 | 0.97161 |
| country=India|src=3 | 164726 | 423525 | 2.022 | 0.73 | 0.71 | 0.98656 | 0.98657 | +0.00001 | 0.99693 | 0.99674 | 0.96074 | 0.96114 |
| country=US|src=2 | 247112 | 603618 | 2.022 | 0.73 | 0.73 | 0.99004 | 0.99007 | +0.00003 | 0.99801 | 0.99801 | 0.97357 | 0.97357 |
| country=US|src=3 | 249064 | 634518 | 2.022 | 0.73 | 0.79 | 0.98997 | 0.98999 | +0.00002 | 0.99809 | 0.99840 | 0.97605 | 0.97519 |

Per-fold refits (thresholds chosen without that fold): fold 0: country=India|src=2=0.72, country=India|src=3=0.71, country=US|src=2=0.75, country=US|src=3=0.79; fold 1: country=India|src=2=0.72, country=India|src=3=0.71, country=US|src=2=0.73, country=US|src=3=0.74; fold 2: country=India|src=2=0.72, country=India|src=3=0.71, country=US|src=2=0.76, country=US|src=3=0.79; fold 3: country=India|src=2=0.73, country=India|src=3=0.73, country=US|src=2=0.73, country=US|src=3=0.80

## indic: per-class thresholds and metrics (OOF, in-sample)

| class | entities | link rows | decoy ratio | thr global | thr class | F0.5 global | F0.5 class | delta | P global | P class | R global | R class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| indic=0 | 431856 | 1914466 | 2.022 | 0.73 | 0.73 | 0.98792 | 0.98792 | +0.00001 | 0.99762 | 0.99762 | 0.97099 | 0.97099 |
| indic=1 | 60189 | 150845 | 2.022 | 0.73 | 0.65 | 0.98552 | 0.98557 | +0.00005 | 0.99905 | 0.99895 | 0.97491 | 0.97520 |

Per-fold refits (thresholds chosen without that fold): fold 0: indic=0=0.73, indic=1=0.65; fold 1: indic=0=0.73, indic=1=0.65; fold 2: indic=0=0.73, indic=1=0.64; fold 3: indic=0=0.73, indic=1=0.77

## noaddr: per-class thresholds and metrics (OOF, in-sample)

| class | entities | link rows | decoy ratio | thr global | thr class | F0.5 global | F0.5 class | delta | P global | P class | R global | R class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| noaddr=0 | 436888 | 1997097 | 2.022 | 0.73 | 0.73 | 0.98774 | 0.98774 | +0.00000 | 0.99809 | 0.99809 | 0.98919 | 0.98919 |
| noaddr=1 | 56723 | 68214 | 2.022 | 0.73 | 0.73 | 0.98042 | 0.98042 | +0.00000 | 0.98413 | 0.98413 | 0.57862 | 0.57862 |

Per-fold refits (thresholds chosen without that fold): fold 0: noaddr=0=0.74, noaddr=1=0.76; fold 1: noaddr=0=0.71, noaddr=1=0.76; fold 2: noaddr=0=0.72, noaddr=1=0.73; fold 3: noaddr=0=0.73, noaddr=1=0.77

## country|src|indic: per-class thresholds and metrics (OOF, in-sample)

| class | entities | link rows | decoy ratio | thr global | thr class | F0.5 global | F0.5 class | delta | P global | P class | R global | R class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| country=India|src=2|indic=0 | 141792 | 308555 | 2.022 | 0.73 | 0.72 | 0.98618 | 0.98620 | +0.00002 | 0.99704 | 0.99694 | 0.97035 | 0.97054 |
| country=India|src=2|indic=1 | 54847 | 95095 | 2.022 | 0.73 | 0.63 | 0.98666 | 0.98670 | +0.00004 | 0.99908 | 0.99896 | 0.97508 | 0.97557 |
| country=India|src=3|indic=0 | 156420 | 367775 | 2.022 | 0.73 | 0.71 | 0.98663 | 0.98666 | +0.00003 | 0.99662 | 0.99641 | 0.95865 | 0.95910 |
| country=India|src=3|indic=1 | 37312 | 55750 | 2.022 | 0.73 | 0.77 | 0.98780 | 0.98784 | +0.00004 | 0.99899 | 0.99908 | 0.97463 | 0.97437 |
| country=US|src=2|indic=0 | 247112 | 603618 | 2.022 | 0.73 | 0.73 | 0.99004 | 0.99007 | +0.00003 | 0.99801 | 0.99801 | 0.97357 | 0.97357 |
| country=US|src=3|indic=0 | 249064 | 634518 | 2.022 | 0.73 | 0.79 | 0.98997 | 0.98999 | +0.00002 | 0.99809 | 0.99840 | 0.97605 | 0.97519 |

Per-fold refits (thresholds chosen without that fold): fold 0: country=India|src=2|indic=0=0.72, country=India|src=2|indic=1=0.65, country=India|src=3|indic=0=0.71, country=India|src=3|indic=1=0.77, country=US|src=2|indic=0=0.75, country=US|src=3|indic=0=0.79; fold 1: country=India|src=2|indic=0=0.72, country=India|src=2|indic=1=0.63, country=India|src=3|indic=0=0.71, country=India|src=3|indic=1=0.77, country=US|src=2|indic=0=0.73, country=US|src=3|indic=0=0.74; fold 2: country=India|src=2|indic=0=0.80, country=India|src=2|indic=1=0.63, country=India|src=3|indic=0=0.71, country=India|src=3|indic=1=0.74, country=US|src=2|indic=0=0.76, country=US|src=3|indic=0=0.79; fold 3: country=India|src=2|indic=0=0.73, country=India|src=2|indic=1=0.65, country=India|src=3|indic=0=0.73, country=India|src=3|indic=1=0.77, country=US|src=2|indic=0=0.73, country=US|src=3|indic=0=0.80

## country|src|noaddr: per-class thresholds and metrics (OOF, in-sample)

| class | entities | link rows | decoy ratio | thr global | thr class | F0.5 global | F0.5 class | delta | P global | P class | R global | R class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| country=India|src=2|noaddr=0 | 163047 | 392144 | 2.022 | 0.73 | 0.72 | 0.98620 | 0.98622 | +0.00002 | 0.99782 | 0.99777 | 0.98634 | 0.98646 |
| country=India|src=2|noaddr=1 | 10694 | 11506 | 2.022 | 0.73 | 0.74 | 0.97827 | 0.97832 | +0.00006 | 0.98521 | 0.98547 | 0.59568 | 0.59453 |
| country=India|src=3|noaddr=0 | 163558 | 410741 | 2.022 | 0.73 | 0.71 | 0.98674 | 0.98676 | +0.00002 | 0.99722 | 0.99707 | 0.97781 | 0.97815 |
| country=India|src=3|noaddr=1 | 11426 | 12784 | 2.022 | 0.73 | 0.76 | 0.97712 | 0.97729 | +0.00017 | 0.98468 | 0.98716 | 0.54505 | 0.54121 |
| country=US|src=2|noaddr=0 | 244764 | 581746 | 2.022 | 0.73 | 0.75 | 0.99021 | 0.99024 | +0.00003 | 0.99843 | 0.99851 | 0.99360 | 0.99341 |
| country=US|src=2|noaddr=1 | 19345 | 21872 | 2.022 | 0.73 | 0.76 | 0.98021 | 0.98023 | +0.00002 | 0.98430 | 0.98628 | 0.57942 | 0.57709 |
| country=US|src=3|noaddr=0 | 247048 | 612466 | 2.022 | 0.73 | 0.74 | 0.99013 | 0.99018 | +0.00005 | 0.99852 | 0.99856 | 0.99453 | 0.99444 |
| country=US|src=3|noaddr=1 | 19564 | 22052 | 2.022 | 0.73 | 0.77 | 0.98059 | 0.98065 | +0.00006 | 0.98309 | 0.98620 | 0.58837 | 0.58405 |

Per-fold refits (thresholds chosen without that fold): fold 0: country=India|src=2|noaddr=0=0.72, country=India|src=2|noaddr=1=0.69, country=India|src=3|noaddr=0=0.71, country=India|src=3|noaddr=1=0.64, country=US|src=2|noaddr=0=0.75, country=US|src=2|noaddr=1=0.76, country=US|src=3|noaddr=0=0.79, country=US|src=3|noaddr=1=0.83; fold 1: country=India|src=2|noaddr=0=0.72, country=India|src=2|noaddr=1=0.74, country=India|src=3|noaddr=0=0.71, country=India|src=3|noaddr=1=0.68, country=US|src=2|noaddr=0=0.73, country=US|src=2|noaddr=1=0.76, country=US|src=3|noaddr=0=0.74, country=US|src=3|noaddr=1=0.84; fold 2: country=India|src=2|noaddr=0=0.72, country=India|src=2|noaddr=1=0.71, country=India|src=3|noaddr=0=0.71, country=India|src=3|noaddr=1=0.75, country=US|src=2|noaddr=0=0.76, country=US|src=2|noaddr=1=0.73, country=US|src=3|noaddr=0=0.79, country=US|src=3|noaddr=1=0.77; fold 3: country=India|src=2|noaddr=0=0.73, country=India|src=2|noaddr=1=0.75, country=India|src=3|noaddr=0=0.73, country=India|src=3|noaddr=1=0.76, country=US|src=2|noaddr=0=0.75, country=US|src=2|noaddr=1=0.78, country=US|src=3|noaddr=0=0.80, country=US|src=3|noaddr=1=0.77

