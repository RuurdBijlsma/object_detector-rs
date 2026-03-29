WIP, right now it's not a crate yet, but the code works and produces the same results as the reference python yolo 26e
implementation.

# getting started

* first get the onnx file, by running (working directory = root of the repo):

```bash
uv run .\py-yolo\export_onnx.py
```

Then run:

```bash
cargo run --bin object_detector --profile release
```

### [When using `load-dynamic` feature] ONNX Runtime Library Not Found

OnnxRuntime is dynamically loaded, so if it's not found correctly, then download the correct onnxruntime library
from [GitHub Releases](http://github.com/microsoft/onnxruntime/releases).

Then put the dll/so/dylib location in your `PATH`, or point the `ORT_DYLIB_PATH` env var to it.

**PowerShell example:**

* Adjust path to where the dll is.

```powershell
$env:ORT_DYLIB_PATH = "C:/Apps/onnxruntime/lib/onnxruntime.dll"
```

**Shell example:**

```shell
export ORT_DYLIB_PATH="/usr/local/lib/libonnxruntime.so"
```