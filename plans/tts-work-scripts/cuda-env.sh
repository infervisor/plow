# CUDA-only replica of the plow devShell (no ROCm), built from store paths that
# are already realised. Needed because nix refuses local builds while the stale
# /homeless-shelter exists, and the default shell wants the ROCm TheRock SDK.
S=/nix/store
CUDA=$S/p49i1vrhcaw5nf2r3bwgmwfz5x8zgb14-cuda-merged-12.9
CC15=$S/788mx070y81zjlg5ipcl0cra3afviw9k-gcc-wrapper-15.2.0
CC14=$S/5yxzjciaa3bhizp7z51iy0v27wj3bdjb-gcc-wrapper-14.3.0
export PATH=$S/z480b23kymbmrijrl49246mrzyphli15-cargo-1.95.0/bin:$S/wnhmqix7bippbbzasj29qiyb422g9asg-rustc-wrapper-1.95.0/bin:$S/42vpa7gg09gs6x9v4y7aizmxwmrc35ac-clippy-1.95.0/bin:$CC15/bin:$S/r9941n32g4wyvggz2703dlplbdq8a6rd-cmake-4.1.2/bin:$S/vlq7nnw39j7rwk0pp68w1fcwzpxahm9h-gnumake-4.4.1/bin:$CUDA/bin:$S/9ypz3flqsrl5xl495mm8h645gadjsxi1-coreutils-9.11/bin:$S/gn94gpcp5q08x4v6g8mvw8v4r65rcjzk-gnugrep-3.12/bin:$S/kgxafhycw2kybbqih759ykc2043qyi5j-gnused-4.9/bin:$S/gik3rh1vz2jlgnifb9dh6vc6sxwwz9jj-bash-5.3p9/bin:/nix/var/nix/profiles/default/bin:/usr/bin:/bin
export CUDA_PATH=$CUDA
export PLOW_NVCC=$CUDA/bin/nvcc
export PLOW_NVCC_PATH=$CUDA/bin:$CC14/bin:$S/9ypz3flqsrl5xl495mm8h645gadjsxi1-coreutils-9.11/bin:$S/gn94gpcp5q08x4v6g8mvw8v4r65rcjzk-gnugrep-3.12/bin:$S/kgxafhycw2kybbqih759ykc2043qyi5j-gnused-4.9/bin:$S/gik3rh1vz2jlgnifb9dh6vc6sxwwz9jj-bash-5.3p9/bin
export NVCC_PREPEND_FLAGS="-ccbin $CC14/bin -I $CUDA/include"
export LIBRARY_PATH=$S/chqq8mpmpyfi9kgsngya71akv5xicn03-gcc-15.2.0-lib/lib
# cuBLASLt (decode/prefill library segments) is dlopened by soname from the toolkit.
export LD_LIBRARY_PATH=$S/chqq8mpmpyfi9kgsngya71akv5xicn03-gcc-15.2.0-lib/lib:$CUDA/lib
# Stands in for "inside nix develop" for scripts that test ROCM_PATH; there is no ROCm here.
export ROCM_PATH=/nonexistent-rocm
export CARGO_TARGET_DIR=/root/tts-work/target
export CARGO_HOME=/root/tts-work/cargo-home
