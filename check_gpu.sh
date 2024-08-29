nvcc -o check_gpu -x cu - <<EOF
#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    return 0;
}
EOF

./check_gpu
