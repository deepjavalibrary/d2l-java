#!/usr/bin/env bash

# build_html.sh
# build website based on jupyter notebooks

# ask tablesaw to plot <img> for Sphinx to load
# export D2L_PLOT_IMAGE=1

set -e

rm -r -f */temp.ipynb

output_dir=_build/eval
mkdir -p $output_dir

aws s3 sync s3://d2l-java-resources/d2l-original .
mkdir -p $output_dir/img
cp -r img/* $output_dir/img
cp d2l.bib $output_dir

python3 tools/add_online_runner.py

set +e
echo "Try to fetch backup"
COMMIT_ID=$(git rev-parse --short HEAD)
D2L_LANG="${D2L_LANG:-en}"
if [ -z "$MAX_EPOCH" ] then;
    DATE=$(date '+%Y-%m-%d')
    S3_PREFIX="$D2L_LANG/$DATE"
else
    S3_PREFIX="$D2L_LANG/$COMMIT_ID"
fi
aws s3 sync "s3://d2l-java-notebook/${S3_PREFIX}" .
set -e

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
    jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=5400 --output temp "$1"
    mkdir -p $output_dir/$dir
    mv "$dir/temp.ipynb" "$output_dir/$1"
    aws s3 cp "$output_dir/$1" "s3://d2l-java-notebook/${S3_PREFIX}/$output_dir/$1"
}

for f in **/*.ipynb
do
  eval "$f"
done

rm -r -f */temp.ipynb

# download additional js to be included
curl -O https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js

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

