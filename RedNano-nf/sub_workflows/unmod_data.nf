//import modules
include { GUPPY_basecalling } from '../modules/guppy_basecalling'
include { TOMBO_resquiggle} from '../modules/tombo_resquiggle'
include { MINIMAP2_align} from '../modules/minimap2_align'
include { Extract_feature} from '../modules/extract_feature'
include { Predict } from '../modules/predict'

workflow Process_Unmod_Data{
    take:
    input_genome
    input_genome_bed
    unmod_data

    main:
    //3.2 PrepareGenome
    GUPPY_basecalling( unmod_data )

    //3.3 Tombo-Resquiggle
    TOMBO_resquiggle ( input_genome, unmod_data, GUPPY_basecalling.out.workspace)

    //3.4 Minimap2-map
    MINIMAP2_align( input_genome, unmod_data, GUPPY_basecalling.out.fasta_guppy_fastq, TOMBO_resquiggle.out.fast5_tombo_log)

    //3.5 Extract-feature
    Extract_feature( unmod_data, MINIMAP2_align.out.fast5_guppy_tsv, input_genome_bed, GUPPY_basecalling.out.workspace)
    
    //3.6 predict
    Predict( unmod_data, Extract_feature.out.fast5_extract_feature_tsv )

    fasta_unmod_5mer_65_rl_pred_tsv = Predict.out.fasta_5mer_65_rl_pred_tsv

    emit:
    fasta_unmod_5mer_65_rl_pred_tsv
}
