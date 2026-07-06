
HFHUB_BASE_URL = "https://huggingface.co/nipponjo/selib-models"

MODEL_URLS = {
    "deepfilternet1": {
        "kind": "deepfilter",
        "url": "deepfilter/DeepFilterNet_v1b.onnx",
        "sha256": "bb6c26890d64f39b5a1851015936f41a3b5bf19c99863af2fcf5eff20a9c2956",
        "license": "MIT OR Apache-2.0",
    },
    "deepfilternet2": {
        "kind": "deepfilter",
        "url": "deepfilter/DeepFilterNet_v2b.onnx",
        "sha256": "e30797232e1a9075dfbc91886301a4274b117d7d84689896b991ee658dccc395",
        "license": "MIT OR Apache-2.0",
    },
    "deepfilternet3": {
        "kind": "deepfilter",
        "url": "deepfilter/DeepFilterNet_v3b.onnx",
        "sha256": "5364e434419f693962433d96566ec8903b16309323a73bc8220baf48546a6733",
        "license": "MIT OR Apache-2.0",
    },
    "ul_unas_16k": {
        "kind": "magnitude_mask",
        "url": "ul-unas/ul_unas_16k.onnx",
        "sha256": "ee2bbc817a2135a89c675fec23d9337638fae23ceb191fabc31ae231d11dcfcd",
        "license": "MIT",
    },
}
