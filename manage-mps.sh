#!/bin/bash

: ${MPSBASE:="/tmp/nvidia-mps-$USER"}
: ${MPSCONTROL:=nvidia-cuda-mps-control}

start()
{
  if pgrep -u "$UID" -fx "$MPSCONTROL -d" >/dev/null; then
    echo "CUDA MPS already running"
    exit 1
  fi

  CUDA_DEVICES=$@

  if [ -z "$CUDA_DEVICES" ]; then
    N_TESLA_DEVICES=$(nvidia-smi -L | grep Tesla | wc -l)
    CUDA_DEVICES="$(seq 0 $[$N_TESLA_DEVICES - 1])"
  fi

  MPSDIRS="$(mktemp -d "$MPSBASE-XXXXXX")"

  declare -i j=0

  for i in $CUDA_DEVICES; do
    D="$MPSDIRS/$j"
    j=$[$j + 1]
    mkdir "$D"
    CUDA_VISIBLE_DEVICES=$i \
      CUDA_MPS_PIPE_DIRECTORY="$D" \
      CUDA_MPS_LOG_DIRECTORY="$D" \
      "$MPSCONTROL" -d
  done
}

findmps()
{
  local MPS=$(pgrep -u "$UID" -fx "$MPSCONTROL -d")
  if [ -z "$MPS" ]; then
    echo "No MPS running"
    exit 1
  fi
  local MPSPIPEDIR=$(tr '\0' '\n' </proc/$MPS/environ |
    sed -n 's/^CUDA_MPS_PIPE_DIRECTORY=\(.*\)/\1/p')
  MPSDIRS=$(dirname $MPSPIPEDIR)
}

stop()
{
  findmps
  for i in "$MPSDIRS"/*; do
    CUDA_MPS_PIPE_DIRECTORY="$i" "$MPSCONTROL" <<<quit
  done
  rm -rf "$MPSDIRS"
}

wrap()
{
  findmps
  N=$(ls "$MPSDIRS/" | wc -l)
  I=${MV2_COMM_WORLD_LOCAL_RANK-${OMPI_COMM_WORLD_LOCAL_RANK-0}}
  export CUDA_MPS_PIPE_DIRECTORY="$MPSDIRS/$[$I % $N]"
  exec "$@"
}

command="$1"
shift

case "$command" in
  "start" | "stop" | "wrap")
    "$command" "$@"
    ;;
  *)
    echo "usage: $0 {start [DEVICE...] | stop | wrap COMMAND}"
esac
