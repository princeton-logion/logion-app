"""
Custom PyInstaller hook for transformers v5
"""
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

module_collection_mode = "py"
hiddenimports = collect_submodules("transformers")
datas = collect_data_files("transformers")
datas += copy_metadata("transformers", recursive=True)