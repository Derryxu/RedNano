# test, for workflow =====================
# @10.51 ===========================

# ----- 1. install
# - rednaon
#git clone https://github.com/Derryxu/RedNano.git  # in ~/tools/
#conda env create -f RedNano/RedNano.yml
#conda activate RedNano
# add rednano to PYTHONPATH
#export PYTHONPATH="${PYTHONPATH}:/home/xyj/tools/RedNano/RedNano"  # absolute path

# - guppy
# download ont-guppy_3.1.5_linux64.tar.gz and unzip it (to ~/tools/ maybe)

# # add hdf5 path
# wget https://github.com/nanoporetech/vbz_compression/releases/download/v1.0.1/ont-vbz-hdf-plugin-1.0.1-Linux-x86_64.tar.gz
# tar zxvf ont-vbz-hdf-plugin-1.0.1-Linux-x86_64.tar.gz
# export HDF5_PLUGIN_PATH=/home/nipeng/tools/ont-vbz-hdf-plugin-1.0.1-Linux/usr/local/hdf5/lib  # absolute path

# ----- 2. prepare demo data
cp -r ~/tools/RedNano/demo/ ./
#cp -r ~/warehousse/RedNano/demo/ ./

# ----- 3. prepare genome
# - index genome
echo -e "\033[1;34m---Index genome---\033[0m"
samtools faidx demo/cc.fasta
java -jar ~/tools/RedNano/RedNano/utils/picard.jar CreateSequenceDictionary -R demo/cc.fasta -O demo/cc.fasta.dict 2>/dev/null
echo -e "\033[32mIndex created successfully\n\033[0m"


# ----- 4. run 
echo -e "\033[1;33m---Starting to process mod data---\033[0m"
# --- 4.1 mod data
# - guppy
echo -e "\033[1;34m---Basecalling using guppy---\033[0m"
~/tools/ont-guppy/bin/guppy_basecaller -i demo/fast5_mod -r -s demo/fast5_mod_guppy --fast5_out -c rna_r9.4.1_70bps_hac.cfg --gpu_runners_per_device 2 --chunks_per_runner 2500 --device CUDA:all > demo/fast5_mod_guppy.log
cat demo/fast5_mod_guppy/*.fastq > demo/fast5_mod_guppy.fastq
echo -e "\033[32mBasecalling completed successfully\033[0m"
tail -n 3 demo/fast5_mod_guppy.log | head -n 1
echo ""

# - tombo
echo -e "\033[1;34m---Resquiggle raw signals---\033[0m"
tombo resquiggle --overwrite demo/fast5_mod_guppy/workspace/ demo/cc.fasta --basecall-group Basecall_1D_001 --fit-global-scale --include-event-stdev --corrected-group RawGenomeCorrected_001 --processes 20 > demo/fast5_mod_tombo.log 2>&1
echo -e "\033[32mResquiggle completed successfully\033[0m"
#tail -n 3 demo/fast5_mod_tombo.log | head -n 1
echo ""

# - minimap2
echo -e "\033[1;34m---Map reads to reference transcriptome---\033[0m"
minimap2 -t 30 -ax map-ont demo/cc.fasta demo/fast5_mod_guppy.fastq 2>/dev/null | samtools view -hSb | samtools sort -@ 30 -o demo/fast5_mod_guppy.bam && \
  samtools index demo/fast5_mod_guppy.bam && \
  samtools view -h -F 3844 demo/fast5_mod_guppy.bam | java -jar ~/tools/RedNano/RedNano/utils/sam2tsv.jar -r demo/cc.fasta -o demo/fast5_mod_guppy.tsv
echo -e "\033[32mMapping completed successfully\n\033[0m"

# - extract features
# make tmp file
echo -e "\033[1;34m---Feature extraction---\033[0m"
mkdir demo/fast5_mod_tmp
awk 'NR==1{ h=$0 }NR>1{ print (!a[$2]++? h ORS $0 : $0) > "demo/fast5_mod_tmp/"$1".txt" }' demo/fast5_mod_guppy.tsv

# -
python ~/tools/RedNano/RedNano/scripts/extract_features.py -i demo/fast5_mod_guppy/workspace/ -o demo/fast5_mod_5mer_65_feature.tsv -n 30 -k 5 -s 65 --errors_dir demo/fast5_mod_tmp --corrected_group RawGenomeCorrected_001 -b demo/cc.bed --w_is_dir no > demo/fast5_mod_extract.log 2>&1
echo -e "\033[32mExtraction completed successfully\033[0m"
#tail -n 3 demo/fast5_mod_extract.log | head -n 2
echo ""

# - predcit at read-level
echo -e "\033[1;34m---Predict m6A site---\033[0m"
CUDA_VISIBLE_DEVICES=0 python ~/tools/RedNano/RedNano/scripts/predict_rl.py --input_file demo/fast5_mod_5mer_65_feature.tsv --output_file demo/fast5_mod_5mer_65_rl_pred.tsv --model ~/tools/RedNano/models/rl_syn_model_states.pt --batch_size 512 --seq_lens 5 --signal_lens 65 --hidden_size 128 --embedding_size 4 --model_type comb_basecall_raw --nproc 10 > demo/fast5_mod_5mer_65_pred.log 2>&1
echo -e "\033[32mPrediction completed successfully\033[0m"
#tail -n 2 demo/fast5_mod_5mer_65_pred.log | head -n 1
echo ""







# --- 4.2 unmod data
echo -e "\033[1;33m---Starting to process unmod data---\033[0m"
# - guppy
echo -e "\033[1;34m---Basecalling using guppy---\033[0m"
~/tools/ont-guppy/bin/guppy_basecaller -i demo/fast5_unmod -r -s demo/fast5_unmod_guppy --fast5_out -c rna_r9.4.1_70bps_hac.cfg --gpu_runners_per_device 2 --chunks_per_runner 2500 --device CUDA:all > demo/fast5_unmod_guppy.log
cat demo/fast5_unmod_guppy/*.fastq > demo/fast5_unmod_guppy.fastq
echo -e "\033[32mBasecalling completed successfully\033[0m"
tail -n 3 demo/fast5_unmod_guppy.log | head -n 1
echo ""

# - tombo
echo -e "\033[1;34m---Resquiggle raw signals---\033[0m"
tombo resquiggle --overwrite demo/fast5_unmod_guppy/workspace/ demo/cc.fasta --basecall-group Basecall_1D_001 --fit-global-scale --include-event-stdev --corrected-group RawGenomeCorrected_001 --processes 20 > demo/fast5_unmod_tombo.log 2>&1
echo -e "\033[32mResquiggle completed successfully\033[0m"
#tail -n 3 demo/fast5_unmod_tombo.log | head -n 1
echo ""

# - minimap2
echo -e "\033[1;34m---Map reads to reference transcriptome---\033[0m"
minimap2 -t 30 -ax map-ont demo/cc.fasta demo/fast5_unmod_guppy.fastq 2>/dev/null | samtools view -hSb | samtools sort -@ 30 -o demo/fast5_unmod_guppy.bam && \
  samtools index demo/fast5_unmod_guppy.bam && \
  samtools view -h -F 3844 demo/fast5_unmod_guppy.bam | java -jar /home/nipeng/tools/RedNano/RedNano/utils/sam2tsv.jar -r demo/cc.fasta > demo/fast5_unmod_guppy.tsv
echo -e "\033[32mMapping completed successfully\n\033[0m"

# - extract features
# make tmp file
echo -e "\033[1;34m---Feature extraction---\033[0m"
mkdir demo/fast5_unmod_tmp
awk 'NR==1{ h=$0 }NR>1{ print (!a[$2]++? h ORS $0 : $0) > "demo/fast5_unmod_tmp/"$1".txt" }' demo/fast5_unmod_guppy.tsv

# -
python ~/tools/RedNano/RedNano/scripts/extract_features.py -i demo/fast5_unmod_guppy/workspace/ -o demo/fast5_unmod_5mer_65_feature.tsv -n 30 -k 5 -s 65 --errors_dir demo/fast5_unmod_tmp --corrected_group RawGenomeCorrected_001 -b demo/cc.bed --w_is_dir no > demo/fast5_unmod_extract.log 2>&1
echo -e "\033[32mExtraction completed successfully\033[0m"
#tail -n 3 demo/fast5_unmod_extract.log | head -n 2
echo ""

# - predcit at read-level
echo -e "\033[1;34m---Predict m6A site---\033[0m"
CUDA_VISIBLE_DEVICES=0 python ~/tools/RedNano/RedNano/scripts/predict_rl.py --input_file demo/fast5_unmod_5mer_65_feature.tsv --output_file demo/fast5_unmod_5mer_65_rl_pred.tsv --model ~/tools/RedNano/models/rl_syn_model_states.pt --batch_size 512 --seq_lens 5 --signal_lens 65 --hidden_size 128 --embedding_size 4 --model_type comb_basecall_raw --nproc 10 > demo/fast5_unmod_5mer_65_rl_pred.log 2>&1
echo -e "\033[32mPrediction completed successfully\033[0m"
#tail -n 2 demo/fast5_unmod_5mer_65_rl_pred.log | head -n 1
echo ""



# ----- 5. evaluate
echo -e "\033[1;34m---Evaluate---\033[0m"
python ~/tools/RedNano/scripts/eval_at_read_level.py 2>/dev/null --unmethylated demo/fast5_unmod_5mer_65_rl_pred.tsv --methylated demo/fast5_mod_5mer_65_rl_pred.tsv --motif DRACH > demo/test_eval_rl_drach.log
echo -e "\033[32mEvaluation completed successfully\n\033[0m"

echo -e "\033[1;32mSucceeed.\033[0m"
echo -e "\033[1;32mCompleted at:  $(date +"%Y-%m-%d %H:%M:%S")\033[0m"
echo -e "\033[1;32mThe evaluation results are as follows:\033[0m"
head -n 10 demo/test_eval_rl_drach.log



















