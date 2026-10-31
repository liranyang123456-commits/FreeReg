# Windows build patches for the transfer probes

The transfer probes run the released code of FoundationPose and Endo3R under
Windows. These patches are the local changes that were needed; apply them to
the stated upstream commits.

| Patch | Upstream repository | Commit |
|---|---|---|
| `foundationpose_windows.patch` | https://github.com/NVlabs/FoundationPose | `a1b694b83e633c2cb6115b9063d940a687759392` |
| `endo3r_windows.patch` | https://github.com/wrld/Endo3R | `8145c8e5535d2111974561e7d1cc06f1d008fe5e` |

```
git clone https://github.com/NVlabs/FoundationPose
cd FoundationPose
git checkout a1b694b83e633c2cb6115b9063d940a687759392
git apply <this repository>/patches/foundationpose_windows.patch
```

FoundationPose: MSVC-compatible headers in the CUDA and C++ extensions, C++17,
`int64_t` accessors and `scalar_type()` for current PyTorch, explicit `dim3`
launch arguments, an `FP_ONLY_OBJS` environment variable that restricts the
Linemod objects, and DLL search paths for torch and CUDA in `run_linemod.py`.

Endo3R: `torch.load(..., weights_only=False)`, required by PyTorch 2.6 and later.

The patches contain absolute paths of the machine that ran the probes (the
Eigen include directory and the torch and CUDA DLL directories); adjust them to
the local installation. The probe scripts expect the two repositories under
`E:\SOTA_Methods\` (`FP_DIR` in `scripts/run_fp_on_clinical_synth.py`; the
Endo3R paths in `scripts/run_endo3r_on_clinical_synth.py`).
