process Predict {
    label 'predict'

    //conda (params.enable_conda ? "${projectDir}/environment.yml" : null)

    publishDir "demo/"

    //${fast5_guppy_dir}   demo/fast5_mod_guppy  or   demo/fast5_unmod_guppy
    //${genome_dir}      demo/cc.fasta
    //${fast5_dir}    demo/fast5_mod

    input:
    path(fast5_dir)
    path(fast5_extract_feature_tsv)

    output:
    path("${fast5_dir}_5mer_65_rl_pred.tsv"), emit: fasta_5mer_65_rl_pred_tsv
    path("${fast5_dir}_5mer_65_pred.log"), emit: fasta_5mer_65_pred_log

    script:

    """
    CUDA_VISIBLE_DEVICES=0 python ${params.predict_utils} \
        --input_file ${fast5_extract_feature_tsv} \
        --output_file ${fast5_dir}_5mer_65_rl_pred.tsv \
        --model ${params.rl_syn_model_states_utils} \
        --batch_size 512 --seq_lens 5 \
        --signal_lens 65 --hidden_size 128 \
        --embedding_size 4 --model_type comb_basecall_raw --nproc 10 \
        > ${fast5_dir}_5mer_65_pred.log 2>&1

    """
}