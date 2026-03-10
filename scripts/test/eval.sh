#!/bin/bash
# usage: CUDA_VISIBLE_DEVICES=4 ./scripts/test/eval.sh $version: str $test_year: str $error_tol: int


version=$1
test_year=$2
error_tol=$3
dir=lightning_logs/version_$version/

# clean out
rm -rf test_temp/
rm -rf Results_pred_symlg/

# generate predictions
python scripts/test/test.py $version $test_year
if [ ! -f result.zip ]; then
  echo "Error: result.zip not found after test. Run from project root." >&2
  exit 1
fi

# copy predictions to target folder
cp result.zip $dir/$test_year.zip

# dump predictions to temp
mkdir -p test_temp/result
if command -v unzip &>/dev/null; then
  unzip -q result.zip -d test_temp/result
else
  python -c "import zipfile; zipfile.ZipFile('result.zip').extractall('test_temp/result')"
fi

# convert tex to symlg
tex2symlg test_temp/result test_temp/pred_symlg 2>"$dir/tex2symlg_${test_year}.log" || true

# evaluate two symlg folder
evaluate test_temp/pred_symlg data/$test_year/symLg >/dev/null 2>&1

# extract evaluation result and save to target folder
python scripts/test/extract_exprate.py $error_tol >&1 | tee $dir/$test_year.txt
