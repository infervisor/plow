#include <metal_stdlib>
using namespace metal;

inline float asr_bf(float x) {
    uint u = as_type<uint>(x);
    return as_type<float>((u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u);
}
inline float asr_gelu(float x) {
    // Metal has no erf. Evaluate the erf-form GELU via the A&S 7.1.26 approximation.
    float z=x*M_SQRT1_2_F, t=1.0f/(1.0f+0.3275911f*abs(z));
    float r=(((((1.061405429f*t-1.453152027f)*t)+1.421413741f)*t-0.284496736f)*t+0.254829592f)*t*exp(-z*z);
    return asr_bf(0.5f*x*(z<0?r:2.0f-r));
}

kernel void asr_splice(device const float *audio [[buffer(0)]],
    device const ushort *table [[buffer(1)]], device const uint2 *rows [[buffer(2)]],
    device ushort *out [[buffer(3)]], constant uint *p [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    if(index >= p[0]*p[1]) return;
    uint2 row=rows[index/p[1]];
    uint col=index%p[1];
    if(row.y == 0xffffffffu) out[index]=table[ulong(row.x)*p[1]+col];
    else {
        uint bits=as_type<uint>(audio[ulong(row.y)*p[1]+col]);
        out[index]=ushort((bits+0x7fffu+((bits>>16)&1u))>>16);
    }
}

kernel void asr_conv(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],
    device const float *bias [[buffer(2)]], device float *y [[buffer(3)]],
    constant uint *p [[buffer(4)]], uint index [[thread_position_in_grid]]) {
    uint ci=p[0], co=p[1], f=p[2], t=p[3], fo=(f+1)/2, to=(t+1)/2;
    if(index>=p[4]*co*fo*to) return;
    uint ot=index%to, of=(index/to)%fo, oc=(index/(to*fo))%co, b=index/(to*fo*co);
    float v=0;
    for(uint c=0;c<ci;c++) for(int i=0;i<3;i++) for(int j=0;j<3;j++) {
        int fi=int(of*2)+i-1, ti=int(ot*2)+j-1;
        if(fi>=0 && fi<int(f) && ti>=0 && ti<int(t))
            v += x[((b*ci+c)*f+uint(fi))*t+uint(ti)]*w[((oc*ci+c)*3+uint(i))*3+uint(j)];
    }
    y[index]=asr_gelu(asr_bf(v+bias[oc]));
}

kernel void asr_conv_tiled(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],
    device const float *bias [[buffer(2)]], device float *y [[buffer(3)]],
    constant uint *p [[buffer(4)]], uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint ci=p[0], co=p[1], f=p[2], t=p[3], fo=(f+1)/2, to=(t+1)/2;
    uint spatial=fo*to, mt=(spatial+7)/8, nt=(co+7)/8;
    uint batch=group/(mt*nt), row=((group/nt)%mt)*8, col=(group%nt)*8;
    threadgroup float a_tile[64], b_tile[64], result[64];
    simdgroup_float8x8 acc(0.0f);
    for(uint base=0;base<ci*9;base+=8) {
        for(uint e=lane;e<64;e+=32) {
            uint r=e/8, c=e%8, k=base+c, pos=row+r;
            int fi=int((pos/to)*2)+int((k%9)/3)-1;
            int ti=int((pos%to)*2)+int(k%3)-1;
            a_tile[e]=(pos<spatial && k<ci*9 && fi>=0 && fi<int(f) && ti>=0 && ti<int(t))
                ?x[((batch*ci+k/9)*f+uint(fi))*t+uint(ti)]:0.0f;
            b_tile[e]=(col+r<co && k<ci*9)?w[(col+r)*ci*9+k]:0.0f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 a, b;
        simdgroup_load(a,a_tile,8);
        simdgroup_load(b,b_tile,8,ulong2(0),true);
        simdgroup_multiply_accumulate(acc,a,b,acc);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(acc,result,8);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for(uint e=lane;e<64;e+=32) {
        uint pos=row+e/8, oc=col+e%8;
        if(pos<spatial && oc<co)
            y[(batch*co+oc)*spatial+pos]=asr_gelu(asr_bf(result[e]+bias[oc]));
    }
}

kernel void asr_unfold(device const float *x [[buffer(0)]], device const float *unused1 [[buffer(1)]],
    device const float *unused2 [[buffer(2)]], device float *y [[buffer(3)]],
    constant uint *p [[buffer(4)]], uint index [[thread_position_in_grid]]) {
    uint ci=p[0], f=p[2], t=p[3], fo=(f+1)/2, to=(t+1)/2, k=ci*9;
    if(index>=p[4]*fo*to*k) return;
    uint row=index/k, q=index%k, batch=row/(fo*to), pos=row%(fo*to);
    int fi=int((pos/to)*2)+int((q%9)/3)-1, ti=int((pos%to)*2)+int(q%3)-1;
    y[index]=(fi>=0 && fi<int(f) && ti>=0 && ti<int(t))
        ?x[((batch*ci+q/9)*f+uint(fi))*t+uint(ti)]:0.0f;
}

kernel void asr_unpack_conv(device const float *x [[buffer(0)]], device const float *unused1 [[buffer(1)]],
    device const float *unused2 [[buffer(2)]], device float *y [[buffer(3)]],
    constant uint *p [[buffer(4)]], uint index [[thread_position_in_grid]]) {
    uint co=p[1], spatial=((p[2]+1)/2)*((p[3]+1)/2);
    if(index>=p[4]*co*spatial) return;
    uint pos=index%spatial, oc=(index/spatial)%co, batch=index/(spatial*co);
    y[index]=x[(batch*spatial+pos)*co+oc];
}

kernel void asr_pack(device const float *x [[buffer(0)]], device const float *unused1 [[buffer(1)]],
    device const float *unused2 [[buffer(2)]], device float *y [[buffer(3)]],
    constant uint *p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    uint rows=p[0], width=p[1], t=p[2], f=p[3];
    if(i>=rows*width) return;
    uint row=i/width, col=i%width, b=row/t, ti=row%t;
    y[i]=x[((b*(width/f)+col/f)*f+col%f)*t+ti];
}

kernel void asr_linear(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],
    device const float *bias [[buffer(2)]], device float *y [[buffer(3)]],
    constant uint *p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    uint m=p[0], n=p[1], k=p[2];
    if(i>=m*n) return;
    uint row=i/n, col=i%n;
    float v=0;
    for(uint j=0;j<k;j++) v+=x[row*k+j]*w[col*k+j];
    v=asr_bf(v+(p[3]!=0?bias[col]:0.0f));
    y[i]=p[4]!=0?asr_gelu(v):v;
}

kernel void asr_linear_tiled(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],
    device const float *bias [[buffer(2)]], device float *y [[buffer(3)]],
    constant uint *p [[buffer(4)]], uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint m=p[0], n=p[1], k=p[2], nt=(n+7)/8;
    uint row=(group/nt)*8, col=(group%nt)*8;
    threadgroup float a_tile[64], b_tile[64], result[64];
    simdgroup_float8x8 acc(0.0f);
    for(uint base=0;base<k;base+=8) {
        for(uint e=lane;e<64;e+=32) {
            uint r=e/8, c=e%8;
            a_tile[e]=(row+r<m && base+c<k)?x[(row+r)*k+base+c]:0.0f;
            b_tile[e]=(col+r<n && base+c<k)?w[(col+r)*k+base+c]:0.0f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 a, b;
        simdgroup_load(a,a_tile,8);
        simdgroup_load(b,b_tile,8,ulong2(0),true);
        simdgroup_multiply_accumulate(acc,a,b,acc);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(acc,result,8);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for(uint e=lane;e<64;e+=32) {
        uint r=row+e/8, c=col+e%8;
        if(r<m && c<n) {
            float v=asr_bf(result[e]+(p[3]!=0?bias[c]:0.0f));
            y[r*n+c]=p[4]!=0?asr_gelu(v):v;
        }
    }
}

kernel void asr_linear_wide(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],
    device const float *bias [[buffer(2)]], device float *y [[buffer(3)]],
    constant uint *p [[buffer(4)]], uint group [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint m=p[0], n=p[1], k=p[2], nt=(n+31)/32;
    uint row=(group/nt)*16, col=(group%nt)*32;
    uint sr=(sg/2)*8, sc=(sg%2)*16;
    threadgroup float a_tile[512], b_tile[1024], result[512];
    simdgroup_float8x8 acc0(0.0f), acc1(0.0f);
    for(uint base=0;base<k;base+=32) {
        for(uint e=lid;e<512;e+=128) {
            uint r=e/32, c=e%32;
            a_tile[e]=(row+r<m && base+c<k)?x[(row+r)*k+base+c]:0.0f;
        }
        for(uint e=lid;e<1024;e+=128) {
            uint r=e/32, c=e%32;
            b_tile[e]=(col+r<n && base+c<k)?w[(col+r)*k+base+c]:0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint q=0;q<32;q+=8) {
            simdgroup_float8x8 a, b0, b1;
            simdgroup_load(a,a_tile+sr*32+q,32);
            simdgroup_load(b0,b_tile+sc*32+q,32,ulong2(0),true);
            simdgroup_load(b1,b_tile+(sc+8)*32+q,32,ulong2(0),true);
            simdgroup_multiply_accumulate(acc0,a,b0,acc0);
            simdgroup_multiply_accumulate(acc1,a,b1,acc1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(acc0,result+sg*128,16);
    simdgroup_store(acc1,result+sg*128+8,16);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for(uint e=lane;e<128;e+=32) {
        uint r=row+sr+e/16, c=col+sc+e%16;
        if(r<m && c<n) {
            float v=asr_bf(result[sg*128+e]+(p[3]!=0?bias[c]:0.0f));
            y[r*n+c]=p[4]!=0?asr_gelu(v):v;
        }
    }
}

kernel void asr_linear_large(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],
    device const float *bias [[buffer(2)]], device float *y [[buffer(3)]],
    constant uint *p [[buffer(4)]], uint group [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint m=p[0], n=p[1], k=p[2], nt=(n+63)/64;
    uint row=(group/nt)*32, col=(group%nt)*64;
    uint sr=(sg/4)*16, sc=(sg%4)*16;
    threadgroup float a_tile[1024], b_tile[2048], result[2048];
    simdgroup_float8x8 acc0(0.0f), acc1(0.0f), acc2(0.0f), acc3(0.0f);
    for(uint base=0;base<k;base+=32) {
        for(uint e=lid;e<1024;e+=256) {
            uint r=e/32, c=e%32;
            a_tile[e]=(row+r<m && base+c<k)?x[(row+r)*k+base+c]:0.0f;
        }
        for(uint e=lid;e<2048;e+=256) {
            uint r=e/32, c=e%32;
            b_tile[e]=(col+r<n && base+c<k)?w[(col+r)*k+base+c]:0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint q=0;q<32;q+=8) {
            simdgroup_float8x8 a, a1, b0, b1;
            simdgroup_load(a,a_tile+sr*32+q,32);
            simdgroup_load(a1,a_tile+(sr+8)*32+q,32);
            simdgroup_load(b0,b_tile+sc*32+q,32,ulong2(0),true);
            simdgroup_load(b1,b_tile+(sc+8)*32+q,32,ulong2(0),true);
            simdgroup_multiply_accumulate(acc0,a,b0,acc0);
            simdgroup_multiply_accumulate(acc1,a,b1,acc1);
            simdgroup_multiply_accumulate(acc2,a1,b0,acc2);
            simdgroup_multiply_accumulate(acc3,a1,b1,acc3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(acc0,result+sg*256,16);
    simdgroup_store(acc1,result+sg*256+8,16);
    simdgroup_store(acc2,result+sg*256+128,16);
    simdgroup_store(acc3,result+sg*256+136,16);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for(uint e=lane;e<256;e+=32) {
        uint r=row+sr+e/16, c=col+sc+e%16;
        if(r<m && c<n) {
            float v=asr_bf(result[sg*256+e]+(p[3]!=0?bias[c]:0.0f));
            y[r*n+c]=p[4]!=0?asr_gelu(v):v;
        }
    }
}

kernel void asr_linear_direct(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],
    device const float *bias [[buffer(2)]], device float *y [[buffer(3)]],
    constant uint *p [[buffer(4)]], uint group [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint m=p[0], n=p[1], k=p[2], nt=(n+63)/64;
    uint row=(group/nt)*32, col=(group%nt)*64;
    uint sr=(sg/4)*16, sc=(sg%4)*16;
    threadgroup float a_tile[1024], b_tile[2048];
    simdgroup_float8x8 acc0(0.0f), acc1(0.0f), acc2(0.0f), acc3(0.0f);
    for(uint base=0;base<k;base+=32) {
        for(uint e=lid;e<1024;e+=256) {
            uint r=e/32, c=e%32;
            a_tile[e]=(row+r<m && base+c<k)?x[(row+r)*k+base+c]:0.0f;
        }
        for(uint e=lid;e<2048;e+=256) {
            uint r=e/32, c=e%32;
            b_tile[e]=(col+r<n && base+c<k)?w[(col+r)*k+base+c]:0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint q=0;q<32;q+=8) {
            simdgroup_float8x8 a, a1, b0, b1;
            simdgroup_load(a,a_tile+sr*32+q,32);
            simdgroup_load(a1,a_tile+(sr+8)*32+q,32);
            simdgroup_load(b0,b_tile+sc*32+q,32,ulong2(0),true);
            simdgroup_load(b1,b_tile+(sc+8)*32+q,32,ulong2(0),true);
            simdgroup_multiply_accumulate(acc0,a,b0,acc0);
            simdgroup_multiply_accumulate(acc1,a,b1,acc1);
            simdgroup_multiply_accumulate(acc2,a1,b0,acc2);
            simdgroup_multiply_accumulate(acc3,a1,b1,acc3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint quad=lane/4;
    uint fr=(quad&4)+((lane/2)%4), fc=(quad&2)*2+(lane%2)*2;
    for(uint e=0;e<2;e++) {
        uint r=row+sr+fr, c=col+sc+fc+e;
        if(r<m && c<n) {
            float v=asr_bf(acc0.thread_elements()[e]+(p[3]!=0?bias[c]:0.0f));
            y[r*n+c]=p[4]!=0?asr_gelu(v):v;
        }
        if(r<m && c+8<n) {
            float v=asr_bf(acc1.thread_elements()[e]+(p[3]!=0?bias[c+8]:0.0f));
            y[r*n+c+8]=p[4]!=0?asr_gelu(v):v;
        }
        if(r+8<m && c<n) {
            float v=asr_bf(acc2.thread_elements()[e]+(p[3]!=0?bias[c]:0.0f));
            y[(r+8)*n+c]=p[4]!=0?asr_gelu(v):v;
        }
        if(r+8<m && c+8<n) {
            float v=asr_bf(acc3.thread_elements()[e]+(p[3]!=0?bias[c+8]:0.0f));
            y[(r+8)*n+c+8]=p[4]!=0?asr_gelu(v):v;
        }
    }
}

kernel void asr_linear_tile64(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],
    device const float *bias [[buffer(2)]], device float *y [[buffer(3)]],
    constant uint *p [[buffer(4)]], uint group [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint m=p[0], n=p[1], k=p[2], nt=(n+63)/64;
    uint row=(group/nt)*64, col=(group%nt)*64;
    uint sr=(sg/4)*32, sc=(sg%4)*16;
    threadgroup float a_tile[2048], b_tile[2048];
    simdgroup_float8x8 acc0(0.0f), acc1(0.0f), acc2(0.0f), acc3(0.0f), acc4(0.0f), acc5(0.0f), acc6(0.0f), acc7(0.0f);
    for(uint base=0;base<k;base+=32) {
        for(uint e=lid;e<2048;e+=256) {
            uint r=e/32, c=e%32;
            a_tile[e]=(row+r<m && base+c<k)?x[(row+r)*k+base+c]:0.0f;
        }
        for(uint e=lid;e<2048;e+=256) {
            uint r=e/32, c=e%32;
            b_tile[e]=(col+r<n && base+c<k)?w[(col+r)*k+base+c]:0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint q=0;q<32;q+=8) {
            simdgroup_float8x8 a, a1, a2, a3, b0, b1;
            simdgroup_load(a,a_tile+sr*32+q,32);
            simdgroup_load(a1,a_tile+(sr+8)*32+q,32);
            simdgroup_load(a2,a_tile+(sr+16)*32+q,32);
            simdgroup_load(a3,a_tile+(sr+24)*32+q,32);
            simdgroup_load(b0,b_tile+sc*32+q,32,ulong2(0),true);
            simdgroup_load(b1,b_tile+(sc+8)*32+q,32,ulong2(0),true);
            simdgroup_multiply_accumulate(acc0,a,b0,acc0);
            simdgroup_multiply_accumulate(acc1,a,b1,acc1);
            simdgroup_multiply_accumulate(acc2,a1,b0,acc2);
            simdgroup_multiply_accumulate(acc3,a1,b1,acc3);
            simdgroup_multiply_accumulate(acc4,a2,b0,acc4);
            simdgroup_multiply_accumulate(acc5,a2,b1,acc5);
            simdgroup_multiply_accumulate(acc6,a3,b0,acc6);
            simdgroup_multiply_accumulate(acc7,a3,b1,acc7);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint quad=lane/4;
    uint fr=(quad&4)+((lane/2)%4), fc=(quad&2)*2+(lane%2)*2;
    for(uint e=0;e<2;e++) {
        uint r=row+sr+fr, c=col+sc+fc+e;
        if(r<m && c<n) {
            float v=asr_bf(acc0.thread_elements()[e]+(p[3]!=0?bias[c]:0.0f));
            y[r*n+c]=p[4]!=0?asr_gelu(v):v;
        }
        if(r<m && c+8<n) {
            float v=asr_bf(acc1.thread_elements()[e]+(p[3]!=0?bias[c+8]:0.0f));
            y[r*n+c+8]=p[4]!=0?asr_gelu(v):v;
        }
        if(r+8<m && c<n) {
            float v=asr_bf(acc2.thread_elements()[e]+(p[3]!=0?bias[c]:0.0f));
            y[(r+8)*n+c]=p[4]!=0?asr_gelu(v):v;
        }
        if(r+8<m && c+8<n) {
            float v=asr_bf(acc3.thread_elements()[e]+(p[3]!=0?bias[c+8]:0.0f));
            y[(r+8)*n+c+8]=p[4]!=0?asr_gelu(v):v;
        }
        if(r+16<m && c<n) {
            float v=asr_bf(acc4.thread_elements()[e]+(p[3]!=0?bias[c]:0.0f));
            y[(r+16)*n+c]=p[4]!=0?asr_gelu(v):v;
        }
        if(r+16<m && c+8<n) {
            float v=asr_bf(acc5.thread_elements()[e]+(p[3]!=0?bias[c+8]:0.0f));
            y[(r+16)*n+c+8]=p[4]!=0?asr_gelu(v):v;
        }
        if(r+24<m && c<n) {
            float v=asr_bf(acc6.thread_elements()[e]+(p[3]!=0?bias[c]:0.0f));
            y[(r+24)*n+c]=p[4]!=0?asr_gelu(v):v;
        }
        if(r+24<m && c+8<n) {
            float v=asr_bf(acc7.thread_elements()[e]+(p[3]!=0?bias[c+8]:0.0f));
            y[(r+24)*n+c+8]=p[4]!=0?asr_gelu(v):v;
        }
    }
}

kernel void asr_linear_large_bf16(device const float *x [[buffer(0)]], device const ushort *w [[buffer(1)]],
    device const float *bias [[buffer(2)]], device float *y [[buffer(3)]],
    constant uint *p [[buffer(4)]], uint group [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint m=p[0], n=p[1], k=p[2], nt=(n+63)/64;
    uint row=(group/nt)*32, col=(group%nt)*64;
    uint sr=(sg/4)*16, sc=(sg%4)*16;
    threadgroup float a_tile[1024], b_tile[2048], result[2048];
    simdgroup_float8x8 acc0(0.0f), acc1(0.0f), acc2(0.0f), acc3(0.0f);
    for(uint base=0;base<k;base+=32) {
        for(uint e=lid;e<1024;e+=256) {
            uint r=e/32, c=e%32;
            a_tile[e]=(row+r<m && base+c<k)?x[(row+r)*k+base+c]:0.0f;
        }
        for(uint e=lid;e<2048;e+=256) {
            uint r=e/32, c=e%32;
            b_tile[e]=(col+r<n && base+c<k)?as_type<float>(uint(w[(col+r)*k+base+c])<<16):0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint q=0;q<32;q+=8) {
            simdgroup_float8x8 a, a1, b0, b1;
            simdgroup_load(a,a_tile+sr*32+q,32);
            simdgroup_load(a1,a_tile+(sr+8)*32+q,32);
            simdgroup_load(b0,b_tile+sc*32+q,32,ulong2(0),true);
            simdgroup_load(b1,b_tile+(sc+8)*32+q,32,ulong2(0),true);
            simdgroup_multiply_accumulate(acc0,a,b0,acc0);
            simdgroup_multiply_accumulate(acc1,a,b1,acc1);
            simdgroup_multiply_accumulate(acc2,a1,b0,acc2);
            simdgroup_multiply_accumulate(acc3,a1,b1,acc3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(acc0,result+sg*256,16);
    simdgroup_store(acc1,result+sg*256+8,16);
    simdgroup_store(acc2,result+sg*256+128,16);
    simdgroup_store(acc3,result+sg*256+136,16);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for(uint e=lane;e<256;e+=32) {
        uint r=row+sr+e/16, c=col+sc+e%16;
        if(r<m && c<n) {
            float v=asr_bf(result[sg*256+e]+(p[3]!=0?bias[c]:0.0f));
            y[r*n+c]=p[4]!=0?asr_gelu(v):v;
        }
    }
}

kernel void asr_norm(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],
    device const float *bias [[buffer(2)]], device float *y [[buffer(3)]],
    constant uint *p [[buffer(4)]], uint row [[thread_position_in_grid]]) {
    uint h=p[1]; if(row>=p[0]) return;
    float mean=0, var=0;
    for(uint j=0;j<h;j++) mean+=x[row*h+j];
    mean/=float(h);
    for(uint j=0;j<h;j++) {float d=x[row*h+j]-mean;var+=d*d;}
    float scale=rsqrt(var/float(h)+1e-5f);
    for(uint j=0;j<h;j++) y[row*h+j]=asr_bf((x[row*h+j]-mean)*scale*w[j]+bias[j]);
}

kernel void asr_add(device const float *x [[buffer(0)]], device const float *other [[buffer(1)]],
    device const float *unused [[buffer(2)]], device float *y [[buffer(3)]],
    constant uint *p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if(i>=p[0]*p[1]) return;
    float v;
    if(p[2]!=0) {
        uint h=p[1], col=i%h, time=(i/h)%p[2];
        float angle=float(time)*exp(-log(10000.0f)*float(col%(h/2))/float(h/2-1));
        v=asr_bf(col<h/2?sin(angle):cos(angle));
    } else v=other[i];
    y[i]=asr_bf(x[i]+v);
}

kernel void asr_attention_simd(device const float *q [[buffer(0)]], device const float *k [[buffer(1)]],
    device const float *v [[buffer(2)]], device float *out [[buffer(3)]],
    constant uint *p [[buffer(4)]], uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint rows=p[0], h=p[1], hd=p[2], window=p[3];
    if(hd==0 || h%hd!=0 || window==0 || window>104) return;
    uint heads=h/hd;
    if(group>=rows*heads) return;
    uint row=group/heads, head=group%heads, start=(row/window)*window, end=min(start+window,rows);
    threadgroup float scores[104];
    for(uint j=start+lane;j<end;j+=32) {
        float score=0;
        for(uint d=0;d<hd;d++) score+=q[row*h+head*hd+d]*k[j*h+head*hd+d];
        scores[j-start]=asr_bf(score)*rsqrt(float(hd));
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if(lane==0) {
        float maximum=-INFINITY, sum=0;
        for(uint j=start;j<end;j++) maximum=max(maximum,scores[j-start]);
        for(uint j=start;j<end;j++) {scores[j-start]=exp(scores[j-start]-maximum);sum+=scores[j-start];}
        for(uint j=start;j<end;j++) scores[j-start]=asr_bf(scores[j-start]/sum);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for(uint d=lane;d<hd;d+=32) {
        float value=0;
        for(uint j=start;j<end;j++) value+=scores[j-start]*v[j*h+head*hd+d];
        out[row*h+head*hd+d]=asr_bf(value);
    }
}

kernel void asr_attention(device const float *q [[buffer(0)]], device const float *k [[buffer(1)]],
    device const float *v [[buffer(2)]], device float *out [[buffer(3)]],
    constant uint *p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    uint rows=p[0], h=p[1], hd=p[2], window=p[3];
    if(i>=rows*(h/hd)) return;
    uint row=i/(h/hd), head=i%(h/hd), start=(row/window)*window, end=min(start+window,rows);
    float scores[104], maximum=-INFINITY, sum=0;
    for(uint j=start;j<end;j++) {
        float score=0;
        for(uint d=0;d<hd;d++) score+=q[row*h+head*hd+d]*k[j*h+head*hd+d];
        score=asr_bf(score)*rsqrt(float(hd)); scores[j-start]=score;maximum=max(maximum,score);
    }
    for(uint j=start;j<end;j++) {scores[j-start]=exp(scores[j-start]-maximum);sum+=scores[j-start];}
    for(uint d=0;d<hd;d++) {
        float value=0;
        for(uint j=start;j<end;j++) value+=asr_bf(scores[j-start]/sum)*v[j*h+head*hd+d];
        out[row*h+head*hd+d]=asr_bf(value);
    }
}


kernel void asr_norm_staged(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],
 device const float *bias [[buffer(2)]], device float *y [[buffer(3)]],
 constant uint *p [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
 uint lane [[thread_index_in_threadgroup]]) {
    uint h=p[1]; if(row>=p[0] || h>1024) return;
    threadgroup float values[1024], stats[2];
    for(uint j=lane;j<h;j+=32) values[j]=x[row*h+j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(lane==0) {
        float mean=0, var=0;
        for(uint j=0;j<h;j++) mean+=values[j];
        mean/=float(h);
        for(uint j=0;j<h;j++) {float d=values[j]-mean;var+=d*d;}
        stats[0]=mean; stats[1]=rsqrt(var/float(h)+1e-5f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean=stats[0], scale=stats[1];
    for(uint j=lane;j<h;j+=32) y[row*h+j]=asr_bf((values[j]-mean)*scale*w[j]+bias[j]);
}

kernel void asr_conv_implicit(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],
    device const float *bias [[buffer(2)]], device float *y [[buffer(3)]],
    constant uint *p [[buffer(4)]], uint group [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint ci=p[0], n=p[1], f=p[2], t=p[3], fo=(f+1)/2, to=(t+1)/2;
    uint m=fo*to, k=ci*9, nt=(n+63)/64, mt=(m+31)/32;
    uint batch=group/(mt*nt), row=((group/nt)%mt)*32, col=(group%nt)*64;
    if(batch>=p[4]) return;
    uint sr=(sg/4)*16, sc=(sg%4)*16;
    threadgroup float a_tile[1024], b_tile[2048];
    simdgroup_float8x8 acc0(0.0f), acc1(0.0f), acc2(0.0f), acc3(0.0f);
    for(uint base=0;base<k;base+=32) {
        for(uint e=lid;e<1024;e+=256) {
            uint r=e/32, c=e%32;
            uint pos=row+r, q=base+c;
            int fi=int((pos/to)*2)+int((q%9)/3)-1, ti=int((pos%to)*2)+int(q%3)-1;
            a_tile[e]=(pos<m && q<k && fi>=0 && fi<int(f) && ti>=0 && ti<int(t))
                ?x[((batch*ci+q/9)*f+uint(fi))*t+uint(ti)]:0.0f;
        }
        for(uint e=lid;e<2048;e+=256) {
            uint r=e/32, c=e%32;
            b_tile[e]=(col+r<n && base+c<k)?w[(col+r)*k+base+c]:0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint q=0;q<32;q+=8) {
            simdgroup_float8x8 a, a1, b0, b1;
            simdgroup_load(a,a_tile+sr*32+q,32);
            simdgroup_load(a1,a_tile+(sr+8)*32+q,32);
            simdgroup_load(b0,b_tile+sc*32+q,32,ulong2(0),true);
            simdgroup_load(b1,b_tile+(sc+8)*32+q,32,ulong2(0),true);
            simdgroup_multiply_accumulate(acc0,a,b0,acc0);
            simdgroup_multiply_accumulate(acc1,a,b1,acc1);
            simdgroup_multiply_accumulate(acc2,a1,b0,acc2);
            simdgroup_multiply_accumulate(acc3,a1,b1,acc3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint quad=lane/4;
    uint fr=(quad&4)+((lane/2)%4), fc=(quad&2)*2+(lane%2)*2;
    for(uint e=0;e<2;e++) {
        uint r=row+sr+fr, c=col+sc+fc+e;
        if(r<m && c<n) {
            float v=asr_bf(acc0.thread_elements()[e]+bias[c]);
            y[(batch*n+c)*m+r]=asr_gelu(v);
        }
        if(r<m && c+8<n) {
            float v=asr_bf(acc1.thread_elements()[e]+bias[c+8]);
            y[(batch*n+c+8)*m+r]=asr_gelu(v);
        }
        if(r+8<m && c<n) {
            float v=asr_bf(acc2.thread_elements()[e]+bias[c]);
            y[(batch*n+c)*m+r+8]=asr_gelu(v);
        }
        if(r+8<m && c+8<n) {
            float v=asr_bf(acc3.thread_elements()[e]+bias[c+8]);
            y[(batch*n+c+8)*m+r+8]=asr_gelu(v);
        }
    }
}
