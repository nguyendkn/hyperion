#!/usr/bin/env bash
# Print the YAML of a GPU work pod (and, with `pvc`, of the persistent cache claim) for `kubectl apply -f -`.
#   scripts/pod-spec.sh pvc                    # PersistentVolumeClaim hyperion-cache, 200 GiB (create once; never delete without the owner)
#   scripts/pod-spec.sh pod <gpus> <hours>     # pod hyperion-work with <gpus> GPUs, activeDeadlineSeconds = hours * 3600, /cache mounted
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
      persistentVolumeClaim:
        claimName: hyperion-cache
    - name: shm
      emptyDir:
        medium: Memory
        sizeLimit: 8Gi
YAML
    ;;
  *) echo "usage: scripts/pod-spec.sh pvc | pod <gpus> <hours>" >&2; exit 2 ;;
esac
