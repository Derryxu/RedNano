def helpMessage() {
    log.info"""
    RedNano - Nextflow PIPELINE
    =================================
    Usage:
    The typical command is as follows:
    nextflow run main.nf

    Mandatory arguments:
      --genome        Genome reference .fasta file
      --genome_bed    Genomic features .bed file
      --mod_data      Mod data directory
      --unmod_data    Unmod data directory
    """
}

//import modules
include { Prepare_Genome } from './sub_workflows/prepare_genome'
include { Process_Mod_Data } from './sub_workflows/mod_data'
include { Process_Unmod_Data } from './sub_workflows/unmod_data'
include { Evaluate } from './sub_workflows/evaluate'

workflow {
    //nextflow run main.nf --genome demo/cc.fasta --genome_bed demo/cc.bed --mod_data demo/fast5_mod --unmod_data demo/fast5_unmod
    //--genome      demo/cc.fasta
    //--genome_bed  demo/cc.bed
    //--mod_data    demo/fast5_mod
    //--unmod_data  demo/fast5_unmod

    if ( params.help ){
        helpMessage()
        exit 0 
    }

    input_genome = Channel.fromPath( params.genome, checkIfExists: true)
    mod_data = Channel.fromPath( params.mod_data, checkIfExists: true)
    input_genome_bed = Channel.fromPath( params.genome_bed, checkIfExists: true)
    unmod_data = Channel.fromPath( params.unmod_data, checkIfExists: true)
 
    //1. PrepareGenome
    Prepare_Genome( input_genome )

    //2. starting to process mod data
    Process_Mod_Data( input_genome, input_genome_bed, mod_data)
    
    //3. starting to process unmod data
    Process_Unmod_Data( input_genome, input_genome_bed, unmod_data)

    //4. evaluate
    Evaluate( Process_Unmod_Data.out.fasta_unmod_5mer_65_rl_pred_tsv, Process_Mod_Data.out.fasta_mod_5mer_65_rl_pred_tsv)
}
