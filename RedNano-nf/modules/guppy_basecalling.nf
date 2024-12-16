process GUPPY_basecalling {
    label 'GUPPY_basecalling'

    //conda (params.enable_conda ? "${projectDir}/environment.yml" : null)

    publishDir "demo/"

    //${fast5_dir}    demo/fast5_mod  or   demo/fast5_unmod

    input:
    path(fast5_dir)

    output:
    path("${fast5_dir}_guppy"),emit: guppy_dir
    path("${fast5_dir}_guppy/workspace"), emit:workspace
    //path("${fast5_dir}_guppy/*.fastq"), emit: fastq
    path("${fast5_dir}_guppy.log"), emit: fasta_guppy_log
    path("${fast5_dir}_guppy.fastq"), emit: fasta_guppy_fastq


    script:

    //println(fast5_dir)
    """
    mkdir -p ${fast5_dir}_guppy
    mkdir -p ${fast5_dir}_guppy/workspace
    
    ${params.guppy_basecaller_utils} \
        -i ${fast5_dir} -r \
        -s ${fast5_dir}_guppy \
        --fast5_out \
        -c rna_r9.4.1_70bps_hac.cfg \
        --gpu_runners_per_device 2 \
        --chunks_per_runner 2500 \
        > ${fast5_dir}_guppy.log

    cat ${fast5_dir}_guppy/*.fastq \
        > ${fast5_dir}_guppy.fastq
    
    """
}