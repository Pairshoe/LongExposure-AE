#!/bin/bash

set -x

sh ./scrips/overall-end2end/run.sh
echo "Script overall-end2end finish!"

sh ./scrips/overall-memory/run.sh
echo "Script overall-memory finish!"

sh ./scrips/ablation-attention/run.sh
echo "Script ablation-attention finish!"

sh ./scrips/ablation-breakdown/run.sh
echo "Script ablation-breakdown finish!"

sh ./scrips/ablation-mlp/run.sh
echo "Script ablation-mlp finish!"

sh ./scrips/ablation-operator/run.sh
echo "Script ablation-operator finish!"

sh ./scrips/scale-model/run.sh
echo "Script scale-model finish!"
