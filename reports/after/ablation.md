== baseline: dropping []
== retrieval: dropping ['score', 'rank', 'cos_name', 'cos_addr', 'cos_cross', 't_best', 't_ncand', 't_best_name', 't_best_addr', 't_best_cross', 't_second', 's_ncand', 's_rank', 's_best', 's_ntop1', 'gap_best', 'gap_second', 'gap_name', 'gap_addr', 'gap_cross', 's_gap_best']
== name_string: dropping ['nm_ratio', 'nm_core_ratio', 'nm_tsort', 'nm_tset', 'nm_partial', 'nm_jw', 'nm_part_max', 'nm_part_min', 'cmp_ratio', 'cmp_jw', 'nm_first_ratio']
== name_tokens: dropping ['t_ntok', 's_ntok', 'nm_common', 'nm_jacc', 'nm_cov_s', 'nm_cov_t', 'nm_xt', 'nm_xs', 't_n_namesake']
== domain: dropping ['dom_prefix', 'dom_full', 't_domain']
== address_string: dropping ['ad_ratio', 'ad_tsort', 'ad_tset', 'ad_partial', 'street_ratio', 'street_tset', 'st_ratio', 'st_eq']
== address_tokens: dropping ['t_natok', 's_natok', 'ad_common', 'ad_jacc']
== house_numbers: dropping ['t_nnum', 's_nnum', 'num_common', 'num_first_eq', 'num_frac', 'hn_rel', 'hn_logdiff', 'hn_reldiff', 'hn_s_in_t', 'hn_t_in_s', 'hn_lendiff', 't_n_addr_exact']
== legal_form: dropping ['lg_t', 'lg_s', 'lg_common', 'lg_conflict']
== record_flags: dropping ['t_src', 't_indic', 't_alias', 't_noaddr', 's_noaddr', 't_noname', 't_nlen', 's_nlen']
== crowding: dropping ['s_addr_mult', 's_street_mult']
== stage2: dropping []

| removed group | #feats | macro F0.5 | delta F0.5 | recall | delta recall | precision | thr | fold std |
|---|---|---|---|---|---|---|---|---|
| baseline | 0 | 0.99089 | +0.00000 | 0.97621 | +0.00000 | 0.99836 | 0.75 | 0.00023 |
| retrieval | 21 | 0.99087 | -0.00003 | 0.97634 | +0.00013 | 0.99822 | 0.73 | 0.00024 |
| name_string | 11 | 0.99051 | -0.00039 | 0.97639 | +0.00018 | 0.99784 | 0.68 | 0.00030 |
| name_tokens | 9 | 0.99069 | -0.00021 | 0.97699 | +0.00078 | 0.99774 | 0.65 | 0.00031 |
| domain | 3 | 0.99082 | -0.00008 | 0.97602 | -0.00019 | 0.99835 | 0.75 | 0.00034 |
| address_string | 8 | 0.99060 | -0.00030 | 0.97589 | -0.00032 | 0.99817 | 0.74 | 0.00037 |
| address_tokens | 4 | 0.99083 | -0.00006 | 0.97647 | +0.00026 | 0.99810 | 0.71 | 0.00036 |
| house_numbers | 12 | 0.98989 | -0.00101 | 0.97488 | -0.00132 | 0.99808 | 0.76 | 0.00019 |
| legal_form | 4 | 0.99020 | -0.00070 | 0.97475 | -0.00145 | 0.99833 | 0.77 | 0.00030 |
| record_flags | 8 | 0.99069 | -0.00021 | 0.97683 | +0.00062 | 0.99787 | 0.68 | 0.00033 |
| crowding | 2 | 0.99088 | -0.00002 | 0.97663 | +0.00042 | 0.99805 | 0.70 | 0.00025 |
| stage2 | 1 | 0.99044 | -0.00046 | 0.97414 | -0.00207 | 0.99826 | 0.79 | 0.00021 |
