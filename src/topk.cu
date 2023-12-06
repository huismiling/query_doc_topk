
#include "topk.h"
#include <stdio.h>
#include <thread>
#include <cassert>
#include <atomic>
#include "faiss_source.cuh"

using namespace faiss;
using namespace faiss::gpu;

#define __SUBMIT_CTRL__ 0

#include <chrono>
#include <sys/time.h>

#define PAD_UP(value, padTo) ((value) % (padTo) == 0 ? (value) : ((value) / (padTo) + 1) * (padTo))

typedef uint4 group_t; // uint32_t

#define MAX_QUERY_NUM (16*1024)
#define QUERY_SCORE_OFFSET (128*4*1024+TOPK)
#define QUERY_NUM_PER_THREAD (6)
#define SIZE_SHM_QUERY (1600)
void __global__ docQueryScoringCoalescedMemoryAccessSampleKernel(
        const __restrict__ uint16_t *docs, 
        const uint16_t *doc_lens, const size_t n_docs, 
        const int base_id, const int total_n_docs,
        uint16_t *querys, uint16_t *querys_lens, const uint64_t n_querys, // int query_id, 
        uint64_t *scores) {

    register auto tid = blockIdx.y * blockDim.x + threadIdx.x, tnum = gridDim.y * blockDim.x;

    if (tid >= n_docs) {
        return;
    }
    register int bid = blockIdx.x*QUERY_NUM_PER_THREAD;
    register uint16_t sub_query_len[QUERY_NUM_PER_THREAD] = {0};
    __shared__ uint32_t query_array_on_shm[SIZE_SHM_QUERY*QUERY_NUM_PER_THREAD];
    #pragma unroll
    for(int i=threadIdx.x; i<SIZE_SHM_QUERY*QUERY_NUM_PER_THREAD; i+=blockDim.x)
        query_array_on_shm[i] = 0;
    __syncthreads();
    #pragma unroll
    for(int i=0; i<QUERY_NUM_PER_THREAD; i++){
        auto query_id = bid + i;
        if(query_id >= n_querys) continue;
        sub_query_len[i] = querys_lens[query_id];
        int query_len = sub_query_len[i]; 
        if (threadIdx.x < query_len) {
            uint16_t q_val = querys[query_id * MAX_DOC_SIZE + threadIdx.x];
            uint32_t a_val = 0x1 << (q_val&0x1f);
            int a_idx = q_val >> 5;
            atomicAdd(query_array_on_shm + i*SIZE_SHM_QUERY + a_idx, a_val);
        }
    }
    __syncthreads();
    
    for (uint64_t doc_id = tid; doc_id < n_docs; doc_id += tnum) {
        register int tmp_score[QUERY_NUM_PER_THREAD] = {0};
        register bool no_more_load = false;
        register int doc_len = doc_lens[doc_id];

        const int i_num = MAX_DOC_SIZE / (sizeof(group_t) / sizeof(uint16_t));
        const int j_num = sizeof(group_t) / sizeof(uint16_t);
        register uint32_t q_a_val[QUERY_NUM_PER_THREAD] = {0};
        register int doc_idx = -1;
        #pragma unroll
        for (auto i = 0; i < i_num; i++) {
            if (no_more_load) {
                break;
            }
            register group_t loaded = ((group_t *)(docs+MAX_DOC_SIZE*doc_id))[i]; // tid
            register uint16_t *doc_segment = (uint16_t*)(&loaded);
            const int j_stide = i*j_num;
            for (auto j = 0; j < j_num; j++) {
                if ((j_stide + j)>=doc_len) {
                    no_more_load = true;
                    break;
                    // return;
                }
                register uint16_t v_doc = doc_segment[j];
                uint32_t d_val = 0x1 << (v_doc &0x1f);
                int d_idx = v_doc >> 5;
                if(d_idx != doc_idx){
                    doc_idx = d_idx;
                    for(int k=0; k<QUERY_NUM_PER_THREAD; k++)
                        q_a_val[k] = query_array_on_shm[k*SIZE_SHM_QUERY+d_idx];
                }
                #pragma unroll
                for(int k=0; k<QUERY_NUM_PER_THREAD; k++){
                    tmp_score[k] += ((d_val&q_a_val[k])!=0);
                }
            }
        }
        #pragma unroll
        for(int i=0; i<QUERY_NUM_PER_THREAD; i++){
            uint64_t query_id = bid + i;
            if(query_id >= n_querys) continue;
            uint64_t query_score_offset = query_id*QUERY_SCORE_OFFSET +TOPK;
            int query_len = sub_query_len[i]; 
            register int max_len = max(query_len, doc_len);
            int tmp_score0 = tmp_score[i]*100000 / max_len;
            uint64_t out_score=0;
            ((uint32_t*)(&out_score))[0] = total_n_docs + 1000 - doc_id - base_id;
            ((uint32_t*)(&out_score))[1] = tmp_score0;
            uint64_t out_idx = query_score_offset +doc_id;
            scores[out_idx] = out_score;
        }
    }
}

class Counter {
public:
    Counter() : counter(0) {}
    void increment(int val=1) {
        counter.fetch_add(val, std::memory_order_relaxed);
    }
    uint32_t getValue() const {
        return counter.load();
    }
private:
    std::atomic<uint32_t> counter;
};

void init_device(void *&workspace, void *h_workspace, size_t total_size,
                Counter &count,
                cudaStream_t stream1,cudaStream_t stream2,cudaStream_t& stream3, cudaStream_t& stream4){
    cudaFree(0);
    cudaStreamCreateWithFlags(&stream3, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream4, cudaStreamNonBlocking);
    cudaMalloc(&workspace, total_size);
    count.increment(256);
}

#define HDOCS_NUM_PER_THREAD (128*1024)
void init_hdocs(std::vector<std::vector<uint16_t>> &docs, uint16_t *&d_docs, uint16_t *&h_docs,
                Counter count[], Counter &count_start,
                int start_idx, size_t numThreads, int sp_n_docs){
    auto n_docs = docs.size();
    int st_i=start_idx;
    int k=0;
    int i=0;
    bool stop = false;
    for (; st_i < n_docs; st_i+=numThreads*sp_n_docs) {
        for(k=0; k<sp_n_docs; k++){
            i = st_i + k;
            if( i > n_docs ){
                stop = true;
                break;
            }
            memcpy(h_docs + i*MAX_DOC_SIZE, docs[i].data(), sizeof(uint16_t) * docs[i].size());
        } 
        // printf("sp_n_docs %d, st_i %d, itp %d, k %d \n", sp_n_docs, st_i, st_i/sp_n_docs, k);
        if(stop) break;
        i = st_i + k;
        int idx = (i/sp_n_docs-1);
        count[idx].increment(1);
    }
    if(stop)
    {
        int idx = (i/sp_n_docs);
        count[idx].increment(1);
    }
}

void doc_query_scoring_gpu_function(std::vector<std::vector<uint16_t>> &querys, int n_querys, 
    std::vector<std::vector<uint16_t>> &docs,
    std::vector<uint16_t> &lens,
    std::vector<std::vector<int>> &indices //shape [querys.size(), TOPK]
    ) {
    std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point t3 = std::chrono::high_resolution_clock::now();

    Counter counter[1024];
    Counter counter_start;
    auto n_docs = docs.size();
    const int sp_n_docs = QUERY_SCORE_OFFSET-TOPK;
    const int sp_num = (n_docs+sp_n_docs-1)/sp_n_docs;

    std::vector<int> s_indices(n_docs);
    void *workspace = nullptr;
    cudaStream_t stream1=0;
    cudaStream_t stream2=0;
    cudaStream_t stream3=0;
    cudaStream_t stream4=0;
    char *h_workspace = new char[(size_t)64*1024*1024*1024];
    uint16_t *d_docs = nullptr, *d_query = nullptr;
    uint16_t *d_doc_lens = nullptr, *d_query_lens = nullptr;
    uint64_t *d_scores = nullptr;

    // copy to device
    size_t size_docs = sizeof(uint16_t) * MAX_DOC_SIZE * n_docs;
    size_t size_d_docs = sizeof(uint16_t) * MAX_DOC_SIZE * sp_n_docs *2;
    size_t size_doc_lens = sizeof(uint16_t) * n_docs;
    size_t size_querys = sizeof(uint16_t) * MAX_DOC_SIZE * n_querys;
    size_t size_query_len = sizeof(uint16_t) * n_querys;
    const int max_score_num = QUERY_SCORE_OFFSET;
    size_t size_scores = sizeof(uint64_t) * (max_score_num) * n_querys;

    size_t total_size = 1024*1024 + PAD_UP(size_d_docs, 4*1024) + PAD_UP(size_doc_lens, 4*1024) + 
                    PAD_UP(size_querys, 4*1024) + PAD_UP(size_query_len, 4*1024) + 
                    PAD_UP(size_scores, 4*1024) + 1024*1024;
    std::cout << "n_querys " << n_querys << std::endl;
    std::cout << "total_size " << total_size/1024.0/1024/1024 << " GB " << std::endl;
    std::cout << "size_scores " << size_scores/1024.0/1024/1024 << " GB " << std::endl;

    uint16_t *h_docs      = (uint16_t *)h_workspace;
    uint16_t *h_doc_lens  = (uint16_t *)((char*)h_docs + PAD_UP(size_docs, 4*1024));
    uint16_t *h_query     = (uint16_t *)((char*)h_doc_lens + PAD_UP(size_doc_lens, 4*1024));
    uint16_t *h_query_lens = (uint16_t *)((char*)h_query + PAD_UP(size_querys, 4*1024));
    std::thread t_init_device([&workspace, &h_workspace, total_size, &counter_start, &stream1, &stream2, &stream3, &stream4](){
        init_device(workspace, (void *)h_workspace, total_size, 
                    counter_start,
                    stream1, stream2, stream3, stream4);
    });
    t_init_device.detach();

    memcpy(h_doc_lens, lens.data(), sizeof(uint16_t) * lens.size());

    #if  __SUBMIT_CTRL__
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "init cost " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms " << std::endl;
    #endif
    int numThreads = 16;
    int start_idx = 0;
    std::thread threads[numThreads];
    int process_num = sp_n_docs; 
    for (int i = 0; i < numThreads; ++i) {
        start_idx = i*process_num;
        if(start_idx>=n_docs) break;
        threads[i] = std::thread([&docs, &d_docs, &h_docs, &counter, &counter_start, start_idx, numThreads, sp_n_docs](){
            init_hdocs(docs, d_docs, h_docs, counter, counter_start, start_idx, numThreads, sp_n_docs);
        });
        threads[i].detach();
    }
    #if  __SUBMIT_CTRL__
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "h2d copy cost " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms " << std::endl;
    #endif
    for(int qi=0; qi<n_querys; qi++) {
        auto query = querys[qi];
        const size_t query_len = query.size();
        h_query_lens[qi] = query_len;
        memcpy(h_query+qi*MAX_DOC_SIZE, query.data(), sizeof(uint16_t) * query_len);
    }

    while((counter_start.getValue()&256) != 256){
    }

    d_docs      = (uint16_t *)workspace;
    d_doc_lens  = (uint16_t *)((char*)d_docs + PAD_UP(size_d_docs, 4*1024));
    d_query     = (uint16_t *)((char*)d_doc_lens + PAD_UP(size_doc_lens, 4*1024));
    d_query_lens = (uint16_t *)((char*)d_query + PAD_UP(size_querys, 4*1024));
    d_scores    = (uint64_t *)((char*)d_query_lens + PAD_UP(size_query_len, 4*1024));
    auto cp_size = PAD_UP(size_doc_lens, 4*1024) + 
                    PAD_UP(size_querys, 4*1024) + PAD_UP(size_query_len, 4*1024);    
    cudaMemcpyAsync(d_doc_lens, h_doc_lens, cp_size, cudaMemcpyHostToDevice, stream4);
    CUDA_TEST_ERROR();
    #if  __SUBMIT_CTRL__
    t1 = std::chrono::high_resolution_clock::now();
    auto time_elaps = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "init indices & query h2d cost " << time_elaps << " ms " << std::endl;
    #endif

    {
        #if  __SUBMIT_CTRL__
        t2 = std::chrono::high_resolution_clock::now();
        #endif

        for(int itp=0; itp<sp_num; itp++){
            while((counter[itp].getValue() & 1) ==0){
                // wait host data processed
                // printf("wait 1ms, itp %d, counter.getValue %d\n", 
                //         itp, counter[itp].getValue());
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            uint16_t *d_docs_now = d_docs + MAX_DOC_SIZE* sp_n_docs * (itp&1);
            int process_num = (itp*sp_n_docs+sp_n_docs) > n_docs ?
                                n_docs-itp*sp_n_docs : sp_n_docs;
            cudaMemcpyAsync(d_docs_now,
                            h_docs + MAX_DOC_SIZE* sp_n_docs * itp,
                            sizeof(uint16_t) * process_num * MAX_DOC_SIZE,
                            cudaMemcpyHostToDevice,
                            stream3);
            cudaStreamSynchronize(stream3); 
            cudaStreamSynchronize(stream4); 
            CUDA_TEST_ERROR();

            int block = N_THREADS_IN_ONE_BLOCK;                             
            dim3 grid((n_querys+QUERY_NUM_PER_THREAD-1)/QUERY_NUM_PER_THREAD, 1);
            docQueryScoringCoalescedMemoryAccessSampleKernel
                <<<grid, block, 0, stream4>>>(
                d_docs_now,
                d_doc_lens + itp*sp_n_docs,
                process_num,
                itp*sp_n_docs,
                n_docs,
                d_query,
                d_query_lens,
                (uint64_t)n_querys,
                d_scores);
            // cudaStreamSynchronize(stream4); 
            CUDA_TEST_ERROR();
            runBlockSelect(d_scores, n_querys, max_score_num,
                d_scores, n_querys, (int)max_score_num, 1, TOPK, stream4);
            // cudaStreamSynchronize(stream4); 
            CUDA_TEST_ERROR();
        }
        uint64_t *h_topk_scores = new uint64_t[n_querys*TOPK];
        for(int itq=0; itq<n_querys; itq++){
            cudaMemcpyAsync(h_topk_scores+itq*TOPK, d_scores +itq*QUERY_SCORE_OFFSET, 
                    sizeof(uint64_t)*TOPK, cudaMemcpyDeviceToHost, stream4);
        }
        cudaStreamSynchronize(stream4); 
        {
            for(int qid=0; qid<n_querys; qid++) {
                std::vector<int> sout_indices(TOPK);
                for (int i = 0; i < TOPK; ++i) {
                    uint64_t sc = *(h_topk_scores +qid*TOPK +i);
                    sout_indices[i] = n_docs+1000-(0xffff'ffff&sc);
                }
                indices.push_back(sout_indices);
            }
        }
        #if  __SUBMIT_CTRL__
        t1 = std::chrono::high_resolution_clock::now();
        auto time_elaps_kernel = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        std::cout << "gpu kernel cost " << time_elaps_kernel << " ms " << std::endl;
        #endif
    }
    cudaFree(workspace);
    cudaStreamDestroy(stream3);
    cudaStreamDestroy(stream4);
    delete [] h_workspace;
}
