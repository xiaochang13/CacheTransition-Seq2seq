#!/bin/bash

JAMR_HOME="${HOME}"/jamr
JAMR_OUTPUT_DIR="${HOME}"/jamr/data # Change this to your JAMR data output directory.
PREPROCESS_INPUT_DIR=./jamr_data
for split in training dev test;
do
    DATA_DIR="${JAMR_OUTPUT_DIR}"
    AMR_FILE="${DATA_DIR}/${split}.txt"
    DEP_FILE="${DATA_DIR}/${split}.txt.snt.deps"
    NER_FILE="${DATA_DIR}/${split}.txt.snt.IllinoisNER"
    CDEC_TOK="${DATA_DIR}/${split}.txt.snt.tok"
    ALIGNED="${DATA_DIR}/${split}.txt.aligned"

    OUTPUT_DIR="${PREPROCESS_INPUT_DIR}/${split}"
    mkdir -p "${OUTPUT_DIR}"
    cp "${AMR_FILE}" "${OUTPUT_DIR}/amr"
    cp "${DEP_FILE}" "${OUTPUT_DIR}/dep"
    cp "${CDEC_TOK}" "${OUTPUT_DIR}/cdec_tok"
    cp "${NER_FILE}" "${OUTPUT_DIR}/ner"
    cat "${ALIGNED}" | grep '^# ::alignments ' | sed 's/^# ::alignments //' | sed 's/ ::annotator Aligner.*$//' > "${OUTPUT_DIR}/alignment"

    python ./data_processing/depTokens.py --input_dir "${OUTPUT_DIR}" --lemma_file ./lemmas/der.lemma
    STAGE1_OUTPUT="${JAMR_HOME}/models/conceptID" # Change this to be the JAMR stage1 output directory

    CATEGORIZED_DIR="${OUTPUT_DIR}"_categorized
    python ./data_processing/prepareTokens.py --task categorize --data_dir "${OUTPUT_DIR}" --phrase_file ./phrases --use_lemma --run_dir "${CATEGORIZED_DIR}" --stats_dir stats --table_dir train_tables --freq_dir frequency_tables --resource_dir ./resources --format jamr > "${split}".jamr.align
    
    ORACLE_OUTPUT_DIR="${OUTPUT_DIR}"_oracle
    python ./oracle/oracle.py --data_dir "${OUTPUT_DIR}" --output_dir "${ORACLE_OUTPUT_DIR}" --cache_size 5
done

for split in dev test;
do
    DATA_DIR="${PREPROCESS_INPUT_DIR}/${split}"
    grep "^Span" "${STAGE1_OUTPUT}/${split}.decode.stage1only.err" > "${DATA_DIR}/conceptID"
    python ./data_processing/prepareTokens.py --task prepare --data_dir ${DATA_DIR} --freq_dir train_tables --output ${DATA_DIR}_cache5_decode_input --cache_size 5 --resource_dir ./resources --format jamr --date_file ./dates --phrase_file ./phrases
done
