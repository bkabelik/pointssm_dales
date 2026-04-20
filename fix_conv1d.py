import re
import glob

files = glob.glob('models/point_ssm/selective_scan_interface.py') + glob.glob('exp/dales/*/code/models/point_ssm/selective_scan_interface.py') + glob.glob('mamba/mamba_ssm/ops/selective_scan_interface.py')

for path in list(set(files)):
    with open(path, 'r') as f:
        text = f.read()

    # forward
    text = text.replace(
        'causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, True)',
        'causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, None, True)'
    )
    text = text.replace(
        'causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, False)',
        'causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, None, False)'
    )

    # backward
    text = text.replace(
        'causal_conv1d_cuda.causal_conv1d_bwd(\n            x, conv1d_weight, conv1d_bias, dconv1d_out, dx, True\n        )',
        'causal_conv1d_cuda.causal_conv1d_bwd(\n            x, conv1d_weight, conv1d_bias, dconv1d_out, None, dx, True\n        )'
    )
    text = text.replace(
        'causal_conv1d_cuda.causal_conv1d_bwd(\n            x, conv1d_weight, conv1d_bias, dconv1d_out, dx, False\n        )',
        'causal_conv1d_cuda.causal_conv1d_bwd(\n            x, conv1d_weight, conv1d_bias, dconv1d_out, None, dx, False\n        )'
    )
    text = text.replace(
        'causal_conv1d_cuda.causal_conv1d_bwd(\n            x, conv1d_weight, conv1d_bias, dconv1d_out, None, True\n        )',
        'causal_conv1d_cuda.causal_conv1d_bwd(\n            x, conv1d_weight, conv1d_bias, dconv1d_out, None, None, True\n        )'
    )
    text = text.replace(
        'causal_conv1d_cuda.causal_conv1d_bwd(\n            x, conv1d_weight, conv1d_bias, dconv1d_out, None, False\n        )',
        'causal_conv1d_cuda.causal_conv1d_bwd(\n            x, conv1d_weight, conv1d_bias, dconv1d_out, None, None, False\n        )'
    )

    with open(path, 'w') as f:
        f.write(text)

print("Replacement complete")
