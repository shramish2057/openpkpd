#!/usr/bin/env bash
set -euo pipefail

# First instantiate the core package
julia -e 'using Pkg; Pkg.activate("packages/core"); Pkg.instantiate()'

# Develop the local OpenPKPDCore package and instantiate CLI dependencies
julia -e '
using Pkg
Pkg.activate("packages/cli")
Pkg.develop(path="packages/core")
Pkg.instantiate()
'

./packages/cli/bin/openpkpd version
./packages/cli/bin/openpkpd replay --artifact validation/golden/pk_iv_bolus.json
./packages/cli/bin/openpkpd validate-golden
