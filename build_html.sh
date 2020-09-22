#!/usr/bin/env bash

# build_html.sh
# build website based on jupyter notebooks

echo "Try to fetch daily backup"
date=$(date '+%Y-%m-%d')
aws s3 sync s3://d2l-java-notebook/daily-backup/$date .

set -e

if [[ $1 == clean ]]; then
    echo "cleaning cached build"
    rm -rf _build
fi

rm -r -f */temp.ipynb

output_dir=_build/eval
mkdir -p $output_dir

aws s3 sync s3://d2l-java-resources/d2l-original .
mkdir -p $output_dir/img
cp -r img/* $output_dir/img
cp d2l.bib $output_dir

d2lbook build eval

function eval {
    base=$(basename $1)
    dir=$(dirname $1)
    if [ -f "$output_dir/$1" ]; then
      echo "$output_dir/$1 exists, skipping."
      return 0
    fi
    echo "Evaluating file: $1"
    echo "saving output to: $output_dir/$1"
    jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=3600 --output temp "$1"
    mkdir -p $output_dir/$dir
    mv "$dir/temp.ipynb" "$output_dir/$1"
    aws s3 cp "$output_dir/$1" "s3://d2l-java-notebook/daily-backup/$date/$output_dir/$1"
}

python3 tools/add_online_runner.py

for f in **/*.ipynb
do
  eval "$f"
done

rm -r -f */temp.ipynb

d2lbook build rst
cp static/frontpage/frontpage.html _build/rst/frontpage.html
mkdir -p _build/rst/chapter_references/
mv zreferences.rst _build/rst/chapter_references/
d2lbook build html
mkdir -p _build/html/_images/
cp -r static/frontpage/_images/* _build/html/_images/
wget https://raw.githubusercontent.com/mli/mx-theme/master/mxtheme/static/sphinx_materialdesign_theme.css.map
wget https://raw.githubusercontent.com/mli/mx-theme/master/mxtheme/static/sphinx_materialdesign_theme.js.map
mv sphinx_materialdesign_theme.css.map _build/html/
mv sphinx_materialdesign_theme.js.map _build/html/

sed -e 's/<blockquote>//g' -i _build/html/index.html
sed -e 's/<\/blockquote>//g' -i _build/html/index.html

for fn in `find _build/html/_images/ -iname '*.svg' `; do
    if [[ $fn == *'qr_'* ]] ; then # || [[ $fn == *'output_'* ]]
        continue
    fi
    # rsvg-convert installed on ubuntu changes unit from px to pt, so evening no
    # change of the size makes the svg larger...
    rsvg-convert -z 1 -f svg -o tmp.svg $fn
    mv tmp.svg $fn
done

