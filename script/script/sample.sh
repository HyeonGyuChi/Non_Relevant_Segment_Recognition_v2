#!/bin/sh

SURGERY='robot' # or 'lapa' or 'robot,lapa'
# surgery type은 해당 수술명에 맞춰 code로 관리
# 학습에 사용할 수술명 코드들을 리스트로 넘겨 parsing
SURGERY_TYPES=['01', '02', ...] 
NRS_COUNTS=10 # 해당 값 이상만 사용
TOTAL_DURATION='30m' # 해당 값 이상만 사용
NRS_DURATION='10s' # 해당 값 이상만 사용


python make_subset.py \
        --surgery ${SURGERY} \
        --surgery_types ${SURGERY_TYPES} \
        --nrs_count ${NRS_COUNTS} \
        --tot_duration ${TOTAL_DURATION} \
        --nrs_duration ${NRS_DURATION} \
        --etc...
