# hybrid-stream-compaction
CUDA implementation of "A Fast Hybrid Approach for Stream Compaction on GPUs" by Rego, Sang and Yu

Made with CUDA v9.1, tested on an NVIDIA GTX 1060 6GB.

There's still something not quite right, as it can handle 128 million elements in about 71ms, instead of the 30ish the article claims to achieve with a Quadro K620.

# Installing
Easiest way is to duplicate the "0_Simple/template" Cuda sample project and just replace all the code with the single .cu file in this repository 

# TODO
* Figure out what is the performance problem
* Cleanup unused includes
