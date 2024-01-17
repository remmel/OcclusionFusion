#!/bin/bash

input_dir="tmp/minion/data"
output_dir="tmp/minion/converted"

mkdir -p "$output_dir/color"
mkdir -p "$output_dir/depth"

for file in "$input_dir"/*.color.png; do
    base_name=$(basename "$file" .color.png)
    base_name=${base_name#"frame-"}

    cp $file "$output_dir/color/${base_name}.png"
done

for file in "$input_dir"/*.depth.png; do
    base_name=$(basename "$file" .depth.png)
    base_name=${base_name#"frame-"}

    cp $file "$output_dir/depth/${base_name}.png"
done
