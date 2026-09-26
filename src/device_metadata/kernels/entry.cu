extern "C" __global__ void watermark_init(const uchar* seed, uchar* output, uint* random,
                                            uint len, uint size, uint bytes, uint mode) {
  uint tid=blockIdx.x*blockDim.x+threadIdx.x;
  wm_initialize(seed,output,random,len,size,bytes,mode,tid);
}
extern "C" __global__ void watermark_repair(const uchar* seed, uchar* output, uint* random,
                                            uint len, uint size, uint bytes, uint mode) {
  uint tid=blockIdx.x*blockDim.x+threadIdx.x;
  wm_repair(seed,random,len,size,tid);
}
extern "C" __global__ void watermark_links(const uchar* seed, uchar* output, uint* random,
                                            uint len, uint size, uint bytes, uint mode) {
  uint tid=blockIdx.x*blockDim.x+threadIdx.x;
  wm_links(random,size,tid);
}
extern "C" __global__ void watermark_parents(const uchar* seed, uchar* output, uint* random,
                                            uint len, uint size, uint bytes, uint mode) {
  uint tid=blockIdx.x*blockDim.x+threadIdx.x;
  wm_parents(random,size,tid);
}
extern "C" __global__ void watermark_resolve(const uchar* seed, uchar* output, uint* random,
                                            uint len, uint size, uint bytes, uint mode) {
  uint tid=blockIdx.x*blockDim.x+threadIdx.x;
  wm_resolve(output,random,size,tid);
}
