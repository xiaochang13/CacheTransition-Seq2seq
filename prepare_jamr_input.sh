#!/bin/bash
. ./scripts/config.sh
# AMR_DIR="${JAMR_HOME}/data/LDC2015E86_DEFT_Phase_2_AMR_Annotation_R1/data/amrs/split"
AMR_DIR="${JAMR_HOME}/data"
JAMR_OUTPUT="${JAMR_HOME}/jamr_data"
for split in training dev test; do
    DATA_DIR="${AMR_DIR}"
    # DATA_DIR="${AMR_DIR}/${split}"
    AMR_FILE="${DATA_DIR}/${split}.txt"
    DEP_FILE="${DATA_DIR}/${split}.txt.snt.deps"
    NER_FILE="${DATA_DIR}/${split}.txt.snt.IllinoisNER"
    CDEC_TOK="${DATA_DIR}/${split}.txt.snt.tok"
    ALIGNED="${DATA_DIR}/${split}.txt.aligned"

    OUTPUT_DIR="${JAMR_OUTPUT}/${split}"
    mkdir -p "${OUTPUT_DIR}"
    cp "${AMR_FILE}" "${OUTPUT_DIR}/amr"
    cp "${DEP_FILE}" "${OUTPUT_DIR}/dep"
    cp "${CDEC_TOK}" "${OUTPUT_DIR}/cdec_tok"
    cp "${NER_FILE}" "${OUTPUT_DIR}/ner"
    cat "${ALIGNED}" | grep '^# ::alignments ' | sed 's/^# ::alignments //' | sed 's/ ::annotator Aligner.*$//' > "${OUTPUT_DIR}/aligned"
done

