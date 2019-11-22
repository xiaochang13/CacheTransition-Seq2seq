#!/bin/bash

JAMR_HOME="${HOME}"/jamr
JAMR_OUTPUT_DIR="${HOME}"/jamr/data # Change this to your JAMR data output directory.
PREPROCESS_INPUT_DIR=./jamr_data
for split in training dev test;
do
    DATA_DIR="${JAMR_OUTPUT_DIR}"
    AMR_FILE="${DATA_DIR}/${split}/${split}.txt"
    DEP_FILE="${DATA_DIR}/${split}/${split}.txt.snt.deps"
    NER_FILE="${DATA_DIR}/${split}/${split}.txt.snt.IllinoisNER"
    CDEC_TOK="${DATA_DIR}/${split}/${split}.txt.snt.tok"
    ALIGNED="${DATA_DIR}/${split}/${split}.txt.aligned"

    OUTPUT_DIR="${PREPROCESS_INPUT_DIR}/${split}"
    mkdir -p "${OUTPUT_DIR}"
    cp "${AMR_FILE}" "${OUTPUT_DIR}/amr"
    cp "${DEP_FILE}" "${OUTPUT_DIR}/dep"
    cp "${CDEC_TOK}" "${OUTPUT_DIR}/cdec_tok"
    cp "${NER_FILE}" "${OUTPUT_DIR}/ner"
    cat "${ALIGNED}" | grep '^# ::alignments ' | sed 's/^# ::alignments //' | sed 's/ ::annotator Aligner.*$//' > "${OUTPUT_DIR}/alignment"

    # using celex lemmatization.
    python ./data_processing/depTokens.py --input_dir "${OUTPUT_DIR}" --lemma_file ./lemmas/der.lemma

    # categorization step.
    python ./data_processing/prepareTokens.py --task categorize --data_dir "${OUTPUT_DIR}" --phrase_file ./phrases --use_lemma --stats_dir stats --table_dir train_tables --freq_dir frequency_tables --resource_dir ./resources --format jamr > "${split}".jamr.align
    
    # oracle action sequence extraction. cache size can be changed.
    ORACLE_OUTPUT_DIR="${OUTPUT_DIR}"_oracle
    python ./oracle/oracle.py --data_dir "${OUTPUT_DIR}" --output_dir "${ORACLE_OUTPUT_DIR}" --cache_size 5
done

for split in dev test;
do
    DATA_DIR="${PREPROCESS_INPUT_DIR}/${split}"

    # use the concept identification results from step1 of JAMR
    STAGE1_OUTPUT="${JAMR_HOME}/models/conceptID" # Change this to be the JAMR stage1 output directory
    grep "^Span" "${STAGE1_OUTPUT}/${split}.decode.stage1only.err" > "${DATA_DIR}/conceptID"

    # prepare the decode input for dev and test.
    python ./data_processing/prepareTokens.py --task prepare --data_dir ${DATA_DIR} --freq_dir train_tables --output ${DATA_DIR}_cache5_decode_input --cache_size 5 --resource_dir ./resources --format jamr --date_file ./dates --phrase_file ./phrases
done
