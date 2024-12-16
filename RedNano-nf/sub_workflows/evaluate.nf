//import modules
include { Evaluation } from '../modules/evaluate'

workflow Evaluate {

    take:
    fast5_unmod_5mer_65_rl_pred_tsv
    fast5_mod_5mer_65_rl_pred_tsv
    
    main:
    Evaluation( fast5_unmod_5mer_65_rl_pred_tsv, fast5_mod_5mer_65_rl_pred_tsv)

}
