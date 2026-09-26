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
extern "C" __global__ void watermark_input(const uchar* seed, const uint* input, uchar* output,
                                           uint len, uint input_len, uint size, uint mode, uint param, uint repeat) {
  wm_input(seed,input,output,len,input_len,size,mode,param,repeat,blockIdx.x*blockDim.x+threadIdx.x);
}
