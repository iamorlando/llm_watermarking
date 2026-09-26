// Shared CUDA/Metal integer implementation. SHA-256 byte order, counter order,
// rejection sampling and Fisher-Yates order match common.rs exactly.
WM_CONSTANT uint WM_K[64] = {
  0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
  0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
  0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
  0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
  0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
  0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
  0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
  0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

WM_FN uint wm_rotr(uint x, uint n) { return (x >> n) | (x << (32-n)); }
WM_FN uint wm_swap(uint x) {
  return (x >> 24) | ((x >> 8) & 0xff00) | ((x << 8) & 0xff0000) | (x << 24);
}

// kind: 0 = SynthID token, 1 = EXP token/tag/counter, 2 = permutation tag/counter.
WM_FN void wm_digest(WM_DEVICE const uchar* seed, uint seed_len, uint kind,
                     wm_u64 index, WM_PRIVATE uint* h) {
  h[0]=0x6a09e667; h[1]=0xbb67ae85; h[2]=0x3c6ef372; h[3]=0xa54ff53a;
  h[4]=0x510e527f; h[5]=0x9b05688c; h[6]=0x1f83d9ab; h[7]=0x5be0cd19;
  wm_u64 len = (wm_u64)seed_len + (kind == 0 ? 4 : kind == 1 ? 13 : 9);
  wm_u64 padded = ((len + 9 + 63) / 64) * 64;
  for (wm_u64 block = 0; block < padded; block += 64) {
    uint w[64];
    for (uint j=0; j<16; ++j) {
      uint word=0;
      for (uint k=0; k<4; ++k) {
        wm_u64 p=block+4*j+k;
        uint b=0;
        if (p < seed_len) b=seed[p];
        else if (p < len) {
          uint s=(uint)(p-seed_len);
          if (kind==0 || kind==1) {
            if (s<4) b=(uint)((index >> (8*s)) & 255);
            else if (s==4) b=2; // EXP tag, followed by counter zero.
          } else if (s>0) b=(uint)((index >> (8*(s-1))) & 255);
        } else if (p==len) b=128;
        else if (p>=padded-8) b=(uint)(((len*8) >> (8*(padded-1-p))) & 255);
        word=(word<<8)|b;
      }
      w[j]=word;
    }
    for (uint j=16; j<64; ++j) {
      uint a=w[j-15], b=w[j-2];
      w[j]=w[j-16]+(wm_rotr(a,7)^wm_rotr(a,18)^(a>>3))+w[j-7]+(wm_rotr(b,17)^wm_rotr(b,19)^(b>>10));
    }
    uint a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],z=h[7];
    for (uint j=0; j<64; ++j) {
      uint t1=z+(wm_rotr(e,6)^wm_rotr(e,11)^wm_rotr(e,25))+((e&f)^((~e)&g))+WM_K[j]+w[j];
      uint t2=(wm_rotr(a,2)^wm_rotr(a,13)^wm_rotr(a,22))+((a&b)^(a&c)^(b&c));
      z=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=z;
  }
}

WM_FN void wm_initialize(WM_DEVICE const uchar* seed, WM_DEVICE uchar* output,
                        WM_DEVICE uint* random, uint len, uint size, uint bytes, uint mode, uint tid) {
  if (tid>=size) return;
  uint h[8];
  if (mode==2) {
    wm_u64 base=(((wm_u64)size+3)/4)*8;
    random[base+size+tid]=0xffffffffu; // Empty linked-list head.
    if ((wm_u64)tid < ((wm_u64)size+3)/4) {
      wm_digest(seed,len,2,tid,h);
      for (uint j=0;j<8;++j) random[(wm_u64)tid*8+j]=wm_swap(h[j]);
      wm_u64 choice_offset=(((wm_u64)size+3)/4)*8;
      for (uint j=0;j<4;++j) {
        wm_u64 cursor=(wm_u64)tid*4+j;
        if (cursor<(wm_u64)size-1) {
          wm_u64 value=(wm_u64)wm_swap(h[j*2]) | ((wm_u64)wm_swap(h[j*2+1])<<32);
          wm_u64 bound=(wm_u64)size-cursor, threshold=((wm_u64)0-bound)%bound;
          // A rejected draw requires repairing the subsequent stream offsets.
          // The flag is device-only, initialized to zero before this pass.
          random[choice_offset+cursor]=(uint)(value%bound);
          if (value<threshold) WM_ATOMIC_OR(random+choice_offset+(wm_u64)size*5,1u);
        }
      }
    }
  } else {
    wm_digest(seed,len,mode,tid,h);
    if (mode==0) {
      for (uint b=0;b<bytes;++b) output[(wm_u64)b*size+tid]=(uchar)(h[b/4] >> (8*(3-b%4)));
    } else {
      wm_u64 word=(wm_u64)wm_swap(h[0]) | ((wm_u64)wm_swap(h[1])<<32);
      wm_u64 n=word>>12;
      // Compute the small tail when U is near one, avoiding F32 rounding to 1.
      // This retains the scalar's open 52-bit half-bin endpoints without F64.
      float cost;
      if (n >= ((wm_u64)1<<51)) {
        float tail=((float)((((wm_u64)1<<52)-1)-n)+0.5f)*0x1p-52f;
        // -ln(1-tail) = 2*atanh(tail/(2-tail)). Here the series argument
        // is <= 1/3; 12 terms reach F32 accuracy without endpoint cancellation
        // or relying on a backend-specific log1p implementation.
        float x=tail/(2.0f-tail), x2=x*x, term=x, sum=x;
        for (uint k=1;k<12;++k) { term*=x2; sum+=term/(float)(2*k+1); }
        cost=2.0f*sum;
      } else {
        float u=((float)n+0.5f)*0x1p-52f;
        cost=-WM_LOG(u);
      }
      ((WM_DEVICE float*)output)[tid]=-WM_LOG(cost);
    }
  }
}

WM_FN wm_u64 wm_draw(WM_DEVICE const uchar* seed, uint len, WM_DEVICE const uint* random,
                    wm_u64 available, wm_u64 cursor) {
  if (cursor<available) return (wm_u64)random[2*cursor] | ((wm_u64)random[2*cursor+1]<<32);
  // Rejections can exhaust the prefetched stream. Continue it exactly on device.
  uint h[8];
  wm_digest(seed,len,2,cursor/4,h);
  uint j=(uint)(cursor%4)*2;
  return (wm_u64)wm_swap(h[j]) | ((wm_u64)wm_swap(h[j+1])<<32);
}

// Almost all streams need no repair: at a 32-bit bound, rejection probability
// per draw is below 2^-32. The rare path still preserves the exact CPU stream.
WM_FN void wm_repair(WM_DEVICE const uchar* seed, WM_DEVICE uint* random,
                    uint len, uint size, uint tid) {
  if (tid!=0) return;
  wm_u64 base=(((wm_u64)size+3)/4)*8;
  if (random[base+(wm_u64)size*5]==0) return;
  wm_u64 cursor=0, available=base/2;
  for (uint i=size-1;i>0;--i) {
    wm_u64 bound=(wm_u64)i+1, threshold=((wm_u64)0-bound)%bound;
    wm_u64 value;
    do { value=wm_draw(seed,len,random,available,cursor++); } while (value<threshold);
    random[base+(size-1-i)]=(uint)(value%bound);
  }
}

// Reconstruct the SAME Fisher-Yates permutation in parallel. A swap at i takes
// the value previously at j[i]. If a higher swap k most recently wrote j[i],
// that value was the value at k before its own swap. This forms an increasing
// parent chain. Lists need not be ordered: taking minima makes the result
// independent of atomic insertion order.
WM_FN void wm_links(WM_DEVICE uint* random, uint size, uint tid) {
  if (tid==0 || tid>=size) return;
  wm_u64 base=(((wm_u64)size+3)/4)*8;
  uint j=random[base+(size-1-tid)];
  random[base+(wm_u64)size*2+tid]=WM_ATOMIC_EXCHANGE(random+base+size+j,tid);
}

WM_FN uint wm_next_writer(WM_DEVICE const uint* random, wm_u64 base, uint size,
                          uint target, uint after) {
  uint node=random[base+size+target], best=0xffffffffu;
  while (node!=0xffffffffu) {
    if (node>after && node<best) best=node;
    node=random[base+(wm_u64)size*2+node];
  }
  return best;
}

WM_FN void wm_parents(WM_DEVICE uint* random, uint size, uint tid) {
  if (tid>=size) return;
  wm_u64 base=(((wm_u64)size+3)/4)*8;
  uint parent=wm_next_writer(random,base,size,tid,tid);
  random[base+(wm_u64)size*3+tid]=parent==0xffffffffu ? tid : parent;
  uint j=tid==0 ? 0 : random[base+(size-1-tid)];
  uint start=wm_next_writer(random,base,size,j,tid);
  // A start <= tid denotes an original token, not a parent to traverse.
  random[base+(wm_u64)size*4+tid]=start==0xffffffffu ? j : start;
}

WM_FN void wm_resolve(WM_DEVICE uchar* output, WM_DEVICE const uint* random, uint size, uint tid) {
  if (tid>=size) return;
  wm_u64 base=(((wm_u64)size+3)/4)*8;
  uint value=random[base+(wm_u64)size*4+tid];
  if (value>tid) {
    uint parent=random[base+(wm_u64)size*3+value];
    while (parent!=value) {
      value=parent;
      parent=random[base+(wm_u64)size*3+value];
    }
  }
  WM_DEVICE uint* order=(WM_DEVICE uint*)output;
  order[tid]=value;
  order[(wm_u64)size+value]=tid;
}
