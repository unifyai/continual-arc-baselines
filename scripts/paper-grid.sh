#!/usr/bin/env bash
# Launch every cell the paper reports, as launch-cell.sh lines read from paper-grid.txt (a floors line launches the
# two floor cells). Each cell runs detached; the paper's grid ran on five 16-vCPU Linux machines, each family of cells
# on a machine of its own, so launch a part at a time on a machine of this size.
#
#   bash scripts/paper-grid.sh [PATTERN]     only the lines matching PATTERN (grep -E), e.g. 'luna6-low' or 'acp .* hermes'
#   DRY_RUN=1 bash scripts/paper-grid.sh     print every cell's run directory and command, launch nothing
#
# Each line of paper-grid.txt is the empty-reply rule its cells ran under (launch-cell.sh, EMPTY_REPLY), then
# launch-cell.sh's arguments.
set -euo pipefail
here=$(dirname "$0")
grep -v '^#' "$here/paper-grid.txt" | grep -E -- "${1:-.}" | while read -r rule args; do
    # shellcheck disable=SC2086  # the arguments are whitespace-separated words
    EMPTY_REPLY=$rule bash "$here/launch-cell.sh" $args
done
