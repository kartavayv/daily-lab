#!/bin/bash

# Get current year, month, and day
YEAR=$(date +%Y)
MONTH=$(date +%m)
DAY=$(date +%d)

# Determine which week of the month it is (based on strict 7-day chunks)
if [ "$DAY" -le 7 ]; then
  WEEK=week1
elif [ "$DAY" -le 14 ]; then
  WEEK=week2
elif [ "$DAY" -le 21 ]; then
  WEEK=week3
elif [ "$DAY" -le 28 ]; then
  WEEK=week4
else
  WEEK=week5
fi

# Folder name like: 2025-06-week2
FOLDER="${YEAR}-${MONTH}-${WEEK}"

# Create and move into the folder
mkdir -p "$FOLDER"
cd "$FOLDER"

# Add README with headings
echo "# ${FOLDER}" > README.md
echo -e "\n## Goals tackled\n\n## Daily Notes\n\n" >> README.md

# Open in VS Code
code .
