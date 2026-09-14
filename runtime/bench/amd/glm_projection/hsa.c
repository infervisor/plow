#include "hsa_backend.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void check(int rc) {
    if (rc) { fprintf(stderr, "%s\n", plow_hsa_last_error()); exit(1); }
}
static void read_into(const char* path, void* dst, size_t bytes) {
    FILE* f = fopen(path, "rb");
    assert(f && fread(dst, 1, bytes, f) == bytes);
    fclose(f);
}
static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
static void put32(uint8_t* args, size_t off, uint32_t value) { memcpy(args + off, &value, 4); }
static void put64(uint8_t* args, size_t off, uint64_t value) { memcpy(args + off, &value, 8); }

int main(int argc, char** argv) {
    if (argc != 13) {
        fprintf(stderr, "usage: hsa OBJECT SYMBOL_FILE M N K MT_I MT_J WG INFO1 A B EXPECTED\n");
        return 2;
    }
    const unsigned m=atoi(argv[3]), n=atoi(argv[4]), k=atoi(argv[5]);
    const unsigned mi=atoi(argv[6]), mj=atoi(argv[7]), wg=atoi(argv[8]), info1=strtoul(argv[9],0,0);
    assert(m && n && k && mi && mj && wg == 256);
    const unsigned grid=((n+mi-1)/mi)*((m+mj-1)/mj);
    FILE* file=fopen(argv[1],"rb"); assert(file);
    fseek(file,0,SEEK_END); size_t bytes=ftell(file); rewind(file);
    void* elf=malloc(bytes); assert(elf && fread(elf,1,bytes,file)==bytes); fclose(file);
    char symbol[4096]={0}; file=fopen(argv[2],"rb"); assert(file);
    size_t slen=fread(symbol,1,sizeof(symbol)-1,file); fclose(file);
    while (slen && (symbol[slen-1]=='\n' || symbol[slen-1]=='\r')) symbol[--slen]=0;
    plow_hsa* h=plow_hsa_init(); assert(h);
    char arch[64]; uint32_t cus,lds;
    check(plow_hsa_device_info(h,0,arch,&cus,&lds));
    assert(!strcmp(arch,"gfx942") && cus==304);
    check(plow_hsa_load_code_object(h,0,elf,bytes)); free(elf);
    plow_hsa_kernel kernel;
    check(plow_hsa_get_kernel(h,0,symbol,&kernel));
    const unsigned descriptor_bytes=kernel.kernarg_size;
    // The inspected assembly notes declare 160 bytes; some descriptors leave this zero.
    assert((descriptor_bytes==0 || descriptor_bytes==160) && kernel.private_segment_size==0);
    kernel.kernarg_size=160; kernel.kernarg_explicit=160;
    const size_t ab=(size_t)m*k*2, bb=(size_t)n*k*2, cb=(size_t)m*n*2, guard=512;
    void* ha=plow_hsa_alloc_host(h,ab); void* hb=plow_hsa_alloc_host(h,bb);
    uint16_t* hc=plow_hsa_alloc_host(h,cb+guard); void* expected=malloc(cb);
    assert(ha && hb && hc && expected);
    read_into(argv[10],ha,ab); read_into(argv[11],hb,bb); read_into(argv[12],expected,cb);
    void* da=plow_hsa_alloc(h,0,ab); void* db=plow_hsa_alloc(h,0,bb);
    void* dc=plow_hsa_alloc(h,0,cb+guard); assert(da && db && dc);
    check(plow_hsa_copy_h2d(h,0,da,ha,ab)); check(plow_hsa_copy_h2d(h,0,db,hb,bb));
    for(size_t i=0;i<(cb+guard)/2;i++) hc[i]=0x7fc1;
    check(plow_hsa_copy_h2d(h,0,dc,hc,cb+guard));
    uint8_t args[160]={0};
    const uint32_t dims[8]={1,1,info1,grid,n,m,1,k}; memcpy(args,dims,32);
    put64(args,32,(uint64_t)dc); put64(args,40,(uint64_t)dc);
    put64(args,48,(uint64_t)db); put64(args,56,(uint64_t)da);
    const uint32_t strides[8]={n,n*m,n,n*m,k,k*n,k,k*m}; memcpy(args+64,strides,32);
    put32(args,96,0x3f800000); put64(args,140,(uint64_t)dc);
    for(unsigned repeat=0;repeat<3;repeat++) {
        check(plow_hsa_launch(h,0,&kernel,grid*wg,1,1,wg,1,1,0,args,sizeof(args)));
        check(plow_hsa_wait(h,0));
        check(plow_hsa_copy_d2h(h,0,hc,dc,cb+guard));
        assert(!memcmp(hc,expected,cb));
        for(size_t i=cb/2;i<(cb+guard)/2;i++) assert(hc[i]==0x7fc1);
    }
    printf("{\"rows\":%u,\"n\":%u,\"k\":%u,\"descriptor_kernarg_bytes\":%u,\"kernarg_bytes\":160,\"lds_bytes\":%u,\"private_bytes\":%u,\"repeats_exact\":3,\"guard_bytes\":%zu,\"samples_ms\":[",m,n,k,descriptor_bytes,kernel.group_segment_size,kernel.private_segment_size,guard);
    for(unsigned sample=0;sample<7;sample++) {
        double begin=now();
        for(unsigned repeat=0;repeat<20;repeat++) check(plow_hsa_launch(h,0,&kernel,grid*wg,1,1,wg,1,1,0,args,sizeof(args)));
        check(plow_hsa_wait(h,0));
        printf("%s%.9f",sample?",":"",(now()-begin)*1000/20);
    }
    printf("]}\n");
    free(expected); plow_hsa_free(h,da); plow_hsa_free(h,db); plow_hsa_free(h,dc);
    plow_hsa_free(h,ha); plow_hsa_free(h,hb); plow_hsa_free(h,hc); plow_hsa_shutdown(h);
    return 0;
}
