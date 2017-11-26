#!/bin/bash
JOB_QUEUE=JobQueue-e06949489e0faa8

RESIZE_JOB_ARN=$(aws batch describe-job-definitions --job-definition-name image-converter-resize-image --status ACTIVE | jq -r '.jobDefinitions | max_by(.revision).jobDefinitionArn')
EDGE_JOB_ARN=$(aws batch describe-job-definitions --job-definition-name image-converter-extract-edge --status ACTIVE | jq -r '.jobDefinitions | max_by(.revision).jobDefinitionArn')

MAKE_SEQ=$(cat <<EOF
for i in range(0, 0x100):
    print("{:0>2x}".format(i))
EOF
        )

SEQ=$(python -c "$MAKE_SEQ")
echo "$SEQ" | xargs -P 8 -I {} -n 1 \
                    ./submit-jobs.sh $JOB_QUEUE $RESIZE_JOB_ARN $EDGE_JOB_ARN {}
