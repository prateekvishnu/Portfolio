#!/bin/bash

# Get the current directory of the script
script_dir=$(dirname "$(readlink -f "$0")")

files=(
  "2011_09_26_calib.zip"
  "2011_09_26_drive_0005"
)

for i in "${files[@]}"; do
  if [[ ${i:(-3)} != "zip" ]]; then
    shortname="$i_sync.zip"
    fullname="$i/$i_sync.zip"
  else
    shortname="$i"
    fullname="$i"
  fi

  echo "Downloading: $shortname"
  wget "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/$fullname" -P "$script_dir"
  unzip -o "$script_dir/$shortname" -d "$script_dir"
  rm "$script_dir/$shortname"
done

