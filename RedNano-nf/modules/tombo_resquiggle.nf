process TOMBO_resquiggle {
    label 'TOMBO_resquiggle'

    //conda (params.enable_conda ? "${projectDir}/environment.yml" : null)

    publishDir "demo/"

    //${fast5_guppy_dir}   demo/fast5_mod_guppy
    //${genome_dir}      demo/cc.fasta
    //${fast5_dir}    demo/fast5_mod

    input:
    //path(fast5_guppy_dir)
    path(genome_dir)
    path(fast5_dir)
    path(workspace)

    output:
    path("${fast5_dir}_tombo.log"), emit: fast5_tombo_log


    script:

    //println(fast5_guppy_dir)
    """
    tombo resquiggle \
        --overwrite ${workspace}/ ${genome_dir} \
        --basecall-group Basecall_1D_001 \
        --fit-global-scale \
        --include-event-stdev \
        --corrected-group RawGenomeCorrected_001 \
        --processes 20 \
        > ${fast5_dir}_tombo.log 2>&1

    """
}