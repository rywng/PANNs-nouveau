#!/usr/bin/env sh
#
# Exports audio that is mis-labeled into a compressed tar file.
#
# To use this script, first run python -m cli.inference, to get the inference output
# Then run this script, with <output>.csv as argument.

result_csv_path=$1

if [ -z $result_csv_path ]; then
	echo "Usage: $0 <inference_output>.csv"
	exit 1
fi

grep humming $result_csv_path | grep ',2' | cut -d "," -f 1 | xargs bsdtar -cLavf humming_fn.tar.zst
grep cough $result_csv_path | grep ',2' | cut -d "," -f 1 | xargs bsdtar -cLavf cough_fn.tar.zst
grep unbalanced $result_csv_path | grep ",[10]" | cut -d "," -f 1 | xargs bsdtar -cLavf fp.tar.zst

echo "Output saved to *.tar.zst"
