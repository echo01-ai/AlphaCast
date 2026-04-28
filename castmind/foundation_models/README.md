# Foundation Model Directory

The current model wrappers look for local foundation models under this legacy
path:

- `castmind/foundation_models/sundial-base-128m`
- `castmind/foundation_models/timesfm`
- `castmind/foundation_models/chronos-bolt-base`

Keep this directory name unless you also update the model wrapper defaults in
`alphacast/models/base.py`.
