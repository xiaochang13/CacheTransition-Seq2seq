#!/bin/bash
#SBATCH -J hard_s --partition=gpu --gres=gpu:1 --time=5-00:00:00 --output=decode.out_uniform --error=decode.err_uniform
#SBATCH --mem=10GB
#SBATCH -c 5
#SBATCH --reservation=xpeng3-20180220

start=`date +%s`
declare -a beam_size=(5 1 20 10) 
for size in ${beam_size[@]}
do
    #python NP2P_beam_decoder.py --model_prefix logs/NP2P.$1 \
    #        --in_path dev_uniform \
    #        --out_path logs/hard_dev_beam${size}.$1\.tok \
    #        --mode beam_decode \
    #        --decode True \
    #        --cache_size 5

    python NP2P_beam_decoder.py --model_prefix logs/NP2P.$1 \
            --in_path test_uniform \
            --out_path logs/hard_test_beam${size}.$1\.tok \
            --mode beam_decode \
            --decode True \
            --cache_size 5
done

end=`date +%s`
runtime=$((end-start))
echo $runtime
