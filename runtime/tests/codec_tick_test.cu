// nvcc -arch=sm_90a -O3 -I runtime/common runtime/tests/codec_tick_test.cu -o codec_tick_test
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../nvidia/op_control_sample.cuh"
#define CK(x) do { auto e = (x); if(e != cudaSuccess) { fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e)); exit(1); } } while(0)
static short pcm16(float x) {
    if (std::isnan(x)) return 0;
    return (short)std::round(std::fmin(1.f,std::fmax(-1.f,x)) * 32767.f);
}
int main(int argc, char** argv) {
    const bool bench = argc > 1;
    unsigned cases = 0;
    for (unsigned cap : {1u,8u,32u,64u,128u,200u,224u,320u}) {
        for (unsigned samples : {7u,1537u,1920u}) {
            const unsigned k = 32, rows = cap > 1 ? cap-1 : cap, slots = cap * 2;
            const unsigned stride = (k*4 + samples*2 + 3) & ~3u;
            std::vector<unsigned> codes(slots*k), map(rows), actual(cap*k+16,0xabababab), expect(cap*k,0);
            std::vector<float> pcm(cap*samples);
            std::vector<unsigned char> output(rows*stride+32,0xab);
            for(unsigned s=0;s<slots;s++) {
                for(unsigned c=0;c<k;c++) codes[s*k+c]=(s*47+c*17)%2048;
                if(s%5==0) {codes[s*k]=4096;for(unsigned c=1;c<k;c++) codes[s*k+c]=0;}
            }
            for(unsigned r=0;r<rows;r++) {
                map[r]=(rows-r-1)*2;
                if(codes[map[r]*k]!=4096)
                    for(unsigned c=0;c<k;c++) expect[r*k+c]=codes[map[r]*k+c];
            }
            const float values[]={NAN,INFINITY,-INFINITY,-2.f,2.f,0.f,-0.f,0.5f,-0.5f,1.f/65534.f,-1.f/65534.f};
            for(size_t i=0;i<pcm.size();i++) pcm[i]=i%3 ? std::sin(float(i)*.003f)*1.2f : values[(i/3)%11];
            unsigned *dc,*dm,*di,*dl;float *dp;unsigned char* out;
            CK(cudaMalloc(&dl,cap*4));CK(cudaMalloc(&dc,codes.size()*4));CK(cudaMalloc(&dm,map.size()*4));CK(cudaMalloc(&di,actual.size()*4));
            CK(cudaMalloc(&dp,pcm.size()*4));CK(cudaMalloc(&out,output.size()));
            CK(cudaMemcpy(dc,codes.data(),codes.size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(dm,map.data(),map.size()*4,cudaMemcpyHostToDevice));
            CK(cudaMemcpy(di,actual.data(),actual.size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(dp,pcm.data(),pcm.size()*4,cudaMemcpyHostToDevice));
            CK(cudaMemcpy(out,output.data(),output.size(),cudaMemcpyHostToDevice));
            tick_codec_codes<<<(cap*k+255)/256,256>>>(dc,dm,di,rows,cap,k,4096);
            tick_codec_lengths<<<(cap+255)/256,256>>>(dc,dm,dl,rows,cap,k,48,4096);
            tick_pack_pcm16<<<rows,256>>>(dc,dm,dp,out+16,rows,k,samples,stride,4096);
            CK(cudaGetLastError());CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(actual.data(),di,actual.size()*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(output.data(),out,output.size(),cudaMemcpyDeviceToHost));
            for(size_t i=0;i<actual.size();i++) if(actual[i]!=(i<expect.size()?expect[i]:0xabababab)) {fprintf(stderr,"code or guard mismatch\n");return 2;}
            std::vector<unsigned> lengths(cap);
            CK(cudaMemcpy(lengths.data(),dl,cap*4,cudaMemcpyDeviceToHost));
            for(unsigned r=0;r<cap;r++) if(lengths[r]!=(r<rows && codes[map[r]*k]!=4096?48u:0u)) return 6;
            for(unsigned i=0;i<16;i++) if(output[i]!=0xab||output[output.size()-1-i]!=0xab) return 3;
            for(unsigned r=0;r<rows;r++) {
                auto* h=(unsigned*)(output.data()+16+r*stride);
                auto* a=(short*)(h+k);
                for(unsigned c=0;c<k;c++) if(h[c]!=codes[map[r]*k+c]) return 4;
                for(unsigned s=0;s<samples;s++) {
                    const short expected=codes[map[r]*k]==4096?0:pcm16(pcm[r*samples+s]);
                    if(a[s]!=expected) {fprintf(stderr,"PCM mismatch cap%u r%u s%u %d != %d\n",cap,r,s,a[s],expected);return 5;}
                }
            }
            if(bench && samples==1920) {
                cudaEvent_t a,b;CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));CK(cudaEventRecord(a));
                for(int i=0;i<500;i++) {
                    tick_codec_codes<<<(cap*k+255)/256,256>>>(dc,dm,di,rows,cap,k,4096);
                    tick_pack_pcm16<<<rows,256>>>(dc,dm,dp,out+16,rows,k,samples,stride,4096);
                }
                CK(cudaEventRecord(b));CK(cudaEventSynchronize(b));float ms;CK(cudaEventElapsedTime(&ms,a,b));
                printf("capacity=%u copy_pack_us=%.3f\n",cap,ms*2);
                CK(cudaEventDestroy(a));CK(cudaEventDestroy(b));
            }
            CK(cudaFree(dl));CK(cudaFree(dc));CK(cudaFree(dm));CK(cudaFree(di));CK(cudaFree(dp));CK(cudaFree(out));++cases;
        }
    }
    printf("PASS codec tick %u cases: sparse slots, stops, padding, int16 edges, guards\n",cases);
}
