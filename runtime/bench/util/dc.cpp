#include <hip/hip_runtime.h>
#include <cstdio>
int main(){int n=0;hipGetDeviceCount(&n);printf("devices=%d\n",n);
for(int i=0;i<n;i++){hipDeviceProp_t p;hipGetDeviceProperties(&p,i);printf("  %d: %s gcn=%s\n",i,p.name,p.gcnArchName);}return 0;}
