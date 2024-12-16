process PrepareGenome {
    label 'PrepareGenome'

    //conda (params.enable_conda ? "${projectDir}/RedNano-main/RedNano.yml" : null)

    publishDir "demo/"

    input:
    path(genome_dir)

    output:
    path("${genome_dir}.fai"), emit: faidx
    path("${genome_dir}.dict"), emit: dict


    script:

    //println(fasta_dir)
    //println(params.jar_utils)
    """
    samtools faidx ${genome_dir}

    java -jar ${params.picard_jar_utils} CreateSequenceDictionary \
         -R ${genome_dir} \
         -O ${genome_dir}.dict
    
    """
}