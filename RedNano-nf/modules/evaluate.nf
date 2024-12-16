process Evaluation {
    label 'evaluate'

    //conda (params.enable_conda ? "${projectDir}/environment.yml" : null)

    publishDir "demo/"

    //${fast5_unmod_5mer_65_rl_pred_tsv}    demo/fast5_unmod_5mer_65_rl_pred_tsv
    //${fast5_mod_5mer_65_rl_pred_tsv}      demo/fast5_mod_5mer_65_rl_pred_tsv

    input:
    path(fast5_unmod_5mer_65_rl_pred_tsv)
    path(fast5_mod_5mer_65_rl_pred_tsv)

    output:
    path("test_eval_rl_drach.log"), emit: test_eval_rl_drach_log

    script:

    println()
    """
    python ${params.eval_at_read_level_utils} \
    --unmethylated ${fast5_unmod_5mer_65_rl_pred_tsv} \
    --methylated ${fast5_mod_5mer_65_rl_pred_tsv} \
    --motif DRACH \
    > test_eval_rl_drach.log

    """
}