#!/usr/bin/env bash
# M3 脚本参数；公共参数由现有 LIGER launcher 统一处理。
copmrec_m3_parse() {
  m3_variant=""
  m3_analysis=""
  m3_sha=""
  m3_checkpoint=""
  m3_args=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --variant|--analysis|--checkpoint-sha256)
        [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || { echo "Error: $1 requires a non-empty value." >&2; return 2; }
        case "$1" in
          --variant) m3_variant="$2" ;;
          --analysis) m3_analysis="$2" ;;
          --checkpoint-sha256) m3_sha="$2" ;;
        esac
        shift 2 ;;
      --variant=*) m3_variant="${1#*=}"; shift ;;
      --analysis=*) m3_analysis="${1#*=}"; shift ;;
      --checkpoint-sha256=*) m3_sha="${1#*=}"; shift ;;
      --checkpoint=*) m3_checkpoint="${1#*=}"; m3_args+=("$1"); shift ;;
      --data-dir|--semantic-id-path|--embedding-path|--checkpoint|--dataset|--devices|--notes|--group|--seed|--master-port|--split)
        [[ $# -ge 2 ]] || { echo "Error: $1 requires a value." >&2; return 2; }
        if [[ "$1" == --checkpoint ]]; then m3_checkpoint="$2"; fi
        m3_args+=("$1" "$2"); shift 2 ;;
      *)
        [[ -n "${1//[[:space:]]/}" ]] || { echo "Error: empty Hydra override." >&2; return 2; }
        m3_args+=("$1"); shift ;;
    esac
  done
}
copmrec_m3_require_variant() {
  case "$m3_variant" in
    no_mixture|no_residual|no_joint_ce) ;;
    *) echo "Error: --variant requires a defined M3 ablation." >&2; return 2 ;;
  esac
}
copmrec_m3_require_sha() {
  [[ "$m3_sha" =~ ^[0-9a-f]{64}$ ]] || { echo "Error: --checkpoint-sha256 requires the audited SHA256." >&2; return 2; }
}
