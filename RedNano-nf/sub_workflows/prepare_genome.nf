//import modules
include { PrepareGenome } from '../modules/prepare_genome'

workflow Prepare_Genome {

    take:
    input_genome
    
    main:
    PrepareGenome( input_genome )

}
