#!/usr/bin/env bash
# Print the YAML of a GPU work pod (and, with `pvc`, of the persistent cache claim) for `kubectl apply -f -`.
#   scripts/pod-spec.sh pvc                    # PersistentVolumeClaim hyperion-cache, 200 GiB (create once; never delete without the owner)
#   scripts/pod-spec.sh pod <gpus> <hours> [ephemeral]
#                                              # pod hyperion-work with <gpus> GPUs, activeDeadlineSeconds = hours * 3600, /cache mounted from the PVC
#                                              # (or, with `ephemeral`, from an emptyDir that dies with the pod: when the PVC cannot be attached)
# The cleanup trap is the caller's: `trap 'kubectl delete pod hyperion-work --ignore-not-found' EXIT`.
set -euo pipefail
case "${1:?pvc | pod <gpus> <hours>}" in
  pvc)
    cat <<YAML
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: hyperion-cache
spec:
  accessModes: [ReadWriteOnce]
  resources:
    requests:
      storage: 200Gi
YAML
    ;;
  pod)
    gpus=${2:?gpus}
    hours=${3:?hours}
    if [ "${4:-}" = ephemeral ]; then
      cache_volume="emptyDir:
        sizeLimit: 150Gi"
    else
      cache_volume="persistentVolumeClaim:
        claimName: hyperion-cache"
    fi
    cat <<YAML
apiVersion: v1
kind: Pod
metadata:
  name: hyperion-work
spec:
  restartPolicy: Never
  activeDeadlineSeconds: $((hours * 3600))
  containers:
    - name: work
      image: nvidia/cuda:12.4.1-devel-ubuntu22.04
      command: ["sleep", "infinity"]
      resources:
        limits:
          nvidia.com/gpu: $gpus
      volumeMounts:
        - name: cache
          mountPath: /cache
        - name: shm
          mountPath: /dev/shm
  volumes:
    - name: cache
      $(printf '%b' "$cache_volume" | sed 's/^/      /;1s/^      //')
    - name: shm
      emptyDir:
        medium: Memory
        sizeLimit: 8Gi
YAML
    ;;
  *) echo "usage: scripts/pod-spec.sh pvc | pod <gpus> <hours>" >&2; exit 2 ;;
esac
