#!/bin/bash

set -x

sh ./scrips/ablation-attention/run.sh

sh ./scrips/ablation-breakdown/run.sh

sh ./scrips/ablation-mlp/run.sh

sh ./scrips/ablation-operator/run.sh

sh ./scrips/ablation-predictor/run.sh

sh ./scrips/overall-end2end/run.sh

sh ./scrips/overall-memory/run.sh

sh ./scrips/scale-model/run.sh
