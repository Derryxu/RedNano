process Extract_feature {
    label 'extract_feature'

    //conda (params.enable_conda ? "${projectDir}/environment.yml" : null)

    publishDir "demo/"

    //${fast5_guppy_dir}   demo/fast5_mod_guppy  or   demo/fast5_unmod_guppy
    //${genome_dir}      demo/cc.fasta
    //${fast5_dir}    demo/fast5_mod

    input:
    path(fast5_dir)
    path(fast5_guppy_tsv)
    path(genome_dir_bed)
    path(workspace)

    output:
    //////path("${fast5_dir}_tmp/\$1.txt" ), emit: fast5_tmp_txt
    path("${fast5_dir}_tmp"), emit:fast5_dir_temp
    path("${fast5_dir}_extract.log" ), emit: fast5_extract_log
    path("${fast5_dir}_5mer_65_feature.tsv" ), emit: fast5_extract_feature_tsv

    script:

    println()
    """
    mkdir ${fast5_dir}_tmp

    awk 'NR==1{ h=\$0 }NR>1{ print (!a[\$2]++? h ORS \$0 :\$0) \
        > "${fast5_dir}_tmp/"\$1".txt" }' ${fast5_guppy_tsv}

    python ${params.extract_features_utils} \
        -i ${workspace} \
        -o ${fast5_dir}_5mer_65_feature.tsv -n 30 -k 5 -s 65 \
        --errors_dir ${fast5_dir}_tmp \
        --corrected_group RawGenomeCorrected_001 -b ${genome_dir_bed} \
        --w_is_dir no \
        > ${fast5_dir}_extract.log 2>&1

    """
}