#!/bin/bash

version=$1

# install lgeval and tex2symlg (in SRSC folder)
export LgEvalDir=$(pwd)/lgeval
export Convert2SymLGDir=$(pwd)/convert2symLG
export PATH=$PATH:$LgEvalDir/bin:$Convert2SymLGDir

years=('2014' '2016' '2019')

for y in "${years[@]}"
do
    echo "**************** start evaluating CROHME $y ****************"
    bash scripts/test/eval.sh $version $y 4
    echo 
done

printf "\nSummary for version_%s:\n" "$version"
printf "%-6s %-12s %-12s %-12s %-12s %-12s\n" "Year" "StructRate" "Exprate0" "Exprate1" "Exprate2" "Exprate3"

for y in "${years[@]}"; do
    f="lightning_logs/version_${version}/${y}.txt"
    if [ -f "$f" ]; then
        struct=$(awk '/Struct Rate/ {print $3}' "$f")
        e0=$(awk '/Exprate 0 tolerated/ {print $4}' "$f")
        e1=$(awk '/Exprate 1 tolerated/ {print $4}' "$f")
        e2=$(awk '/Exprate 2 tolerated/ {print $4}' "$f")
        e3=$(awk '/Exprate 3 tolerated/ {print $4}' "$f")
        printf "%-6s %-12s %-12s %-12s %-12s %-12s\n" "$y" "$struct" "$e0" "$e1" "$e2" "$e3"
    else
        printf "%-6s %-12s %-12s %-12s %-12s %-12s\n" "$y" "NA" "NA" "NA" "NA" "NA"
    fi
done
