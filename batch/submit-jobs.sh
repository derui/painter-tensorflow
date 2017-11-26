#!/bin/bash
JOB_QUEUE=$1
RESIZE_JOB_ARN=$2
EDGE_JOB_ARN=$3
SEQ=$4

now=`date +%s`
resize_job=$(aws batch submit-job --job-name "resize-$SEQ-$now" --job-queue $JOB_QUEUE \
    --job-definition $RESIZE_JOB_ARN \
    --parameters Prefix=full/$SEQ
          )

resize_job_id=$(echo "$resize_job" | jq '.jobId')

aws batch submit-job --job-name "edge-$SEQ-$now" --job-queue $JOB_QUEUE \
    --job-definition $EDGE_JOB_ARN \
    --parameters Prefix=resized/$SEQ \
    --depends-on jobId=$resize_job_id
