#!/bin/bash

# Usage: ./filter.sh <folder_path> <num>

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <folder_path> <num>"
    exit 1
fi

# Assign arguments to variables
folder_path=$1
num=$2

# Check if the folder exists
if [ ! -d "$folder_path" ]; then
    echo "Error: Folder $folder_path does not exist."
    exit 1
fi

# Filter and delete files not starting with a number >= specified num
for file in "$folder_path"/*; do
    filename=$(basename "$file")
    # Extract the leading number (before the first underscore)
    leading_num=$(echo "$filename" | awk -F'_' '{print $1}')
    # Check if leading_num is a number
    if [[ $leading_num =~ ^[0-9]+$ ]]; then
        if (( leading_num >= num )); then
            echo "Keeping: $filename"
        else
            echo "Deleting: $filename"
            rm -f "$file"
        fi
    else
        echo "Skipping (no leading number): $filename"
    fi
done