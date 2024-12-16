process MINIMAP2_align {
    label 'MINIMAP2_align'

    //conda (params.enable_conda ? "${projectDir}/environment.yml" : null)

    publishDir "demo/"

    //${fast5_guppy_dir}   demo/fast5_mod_guppy  or   demo/fast5_unmod_guppy
    //${genome_dir}      demo/cc.fasta
    //${fast5_dir}    demo/fast5_mod

    input:
    path(genome_dir)
    path(fast5_dir)
    path(fast5_guppy_fastq)
    path(control)
    
    output:
    path("${fast5_dir}_guppy.bam"), emit: fast5_guppy_bam
    path("${fast5_dir}_guppy.bam.bai"), emit: fast5_guppy_bam_index
    path("${fast5_dir}_guppy.tsv"), emit: fast5_guppy_tsv

    script:

    """
    minimap2 -t 30 -ax map-ont ${genome_dir} ${fast5_guppy_fastq} | \
        samtools view -hSb | samtools sort -@ 30 -o ${fast5_dir}_guppy.bam && \
        samtools index ${fast5_dir}_guppy.bam && \
        samtools view -h -F 3844 ${fast5_dir}_guppy.bam

    samtools faidx ${genome_dir}

    java -jar ${params.picard_jar_utils} CreateSequenceDictionary \
         -R ${genome_dir} \
         -O ${genome_dir}.dict
    
    java -jar ${params.sam2tsv_jar_utils} \
        -r ${genome_dir} -o ${fast5_dir}_guppy.tsv ${fast5_dir}_guppy.bam

    """
}