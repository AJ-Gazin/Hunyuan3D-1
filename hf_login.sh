#!/bin/bash

# Clear all existing environment variables
for var in $(env | cut -d= -f1); do
    unset "$var"
done

# Preserve essential system variables
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export HOME="${HOME}"
export USER="${USER}"

# Get the RunPod secret and set it as HF_TOKEN
export HF_TOKEN="$(runpodctl get secret HF_TOKEN)"

# Verify the token was set
if [ -z "$HF_TOKEN" ]; then
    echo "Error: Failed to retrieve HF_TOKEN from RunPod secrets"
    exit 1
fi

# Login to huggingface-cli
huggingface-cli login --token "$HF_TOKEN"

# Login to huggingface-hub (via Python)
python3 -c "
from huggingface_hub import login
login('${HF_TOKEN}')
"

# Verify the logins
echo "Verifying huggingface-cli login status:"
huggingface-cli whoami

echo "Script completed successfully"
