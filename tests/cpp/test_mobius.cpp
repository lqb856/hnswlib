#include <gtest/gtest.h>
#include <omp.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <queue>
#include <unordered_set>
#include "hnsw/hnswalg.h"
#include "io.h"


int main(int argc, char** argv) {
  std::string source_path = "";
  std::string query_path = "";

  std::string mode = "";

  if (argc > 1) {
    source_path = std::string(argv[1]);
    query_path = std::string(argv[2]);
    mode = std::string(argv[3]);
  }
  u_int32_t dimension = std::stoi(argv[4]);
  u_int32_t top_k = std::stoi(argv[5]);
  u_int32_t efConstruction = std::stoi(argv[6]);
  u_int32_t M = std::stoi(argv[7]);  
  std::string store_index_path = std::string(argv[8]);
  std::string result_path = std::string(argv[9]);
  benchmark::DataSet dataset;
  benchmark::QuerySet queryset;
  benchmark::IO io;
  io.ReadDataSet(source_path, dataset, dimension);
  io.ReadQuerySet(query_path, queryset, dimension);
  std::cout << "Dataset size: " << dataset._num << std::endl;
  std::cout << "Queryset size: " << queryset._num << std::endl;
  // omp_set_num_threads(16); 
  if (mode == "build") {
    hnswlib::L2Space space(dimension);
    hnswlib::HierarchicalNSW<float>* alg_hnsw =
        new hnswlib::HierarchicalNSW<float>(&space, dataset._num + 1, M,
                                            efConstruction);

    float *mass_trans = new float[(long long)dataset._num * dimension];
    for (long long i = 0; i < dataset._num; ++i) {
      float dist = 0;
      for (long long j = 0; j < dimension; ++j) {
        dist += dataset._vecs[i][j] * dataset._vecs[i][j];
      }
      for (long long j = 0; j < dimension; ++j) {
        mass_trans[i * dimension + j] = dataset._vecs[i][j] / dist;
        // dataset._vecs[i][j] /= dist;
      }
    }
    
    auto zero = std::vector<float>(dimension, 0);
    auto s_insert = std::chrono::system_clock::now();
    alg_hnsw->addPoint(zero.data(), (size_t)dataset._num);

    #pragma omp parallel
    {
      #pragma omp for schedule(dynamic, 16)
      for (size_t i = 0; i < dataset._num; ++i) {
        alg_hnsw->addPoint((void *)(mass_trans + dimension * i), i);
      }
    }
    
    for (size_t i = 0; i < dataset._num; ++i) {
        memcpy(alg_hnsw->getDataByInternalId(i + 1), dataset._vecs[alg_hnsw->getExternalLabel(i + 1)].data(), dimension * sizeof(float));
    }

    alg_hnsw->saveIndex(store_index_path);
    auto e_insert = std::chrono::system_clock::now();
    std::chrono::duration<double> insert_time = e_insert - s_insert;
    std::cout << "Insert time: " << insert_time.count() << std::endl;
  }  else {
    u_int32_t efSearch = std::stoi(argv[10]);
    hnswlib::InnerProductSpace space(dimension);
    hnswlib::HierarchicalNSW<float>* alg_hnsw =
        new hnswlib::HierarchicalNSW<float>(&space, store_index_path, false);
    std::vector<std::vector<std::pair<size_t, float>>> result(queryset._num);
    std::vector<std::vector<u_int32_t>> knns(queryset._num);
    std::vector<std::vector<u_int32_t>> knns_label(queryset._num);
    alg_hnsw->setEf(efSearch);
    auto s_search = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
    for (size_t i = 0; i < queryset._num; ++i) {
      auto query = queryset._queries[i];
      auto query_vec = query._vec;
      auto query_label = query._label;
      auto result = alg_hnsw->searchKnn(query_vec.data(), top_k);
      while (!result.empty()) {
        auto top = result.top();
        knns[i].push_back(top.second);
        knns_label[i].push_back(dataset._label[top.second]);
        result.pop();
      }
      std::reverse(knns[i].begin(), knns[i].end());
      std::reverse(knns_label[i].begin(), knns_label[i].end());
    }
    auto e_search = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> search_time = e_search - s_search;
    std::cout << "search time: " << search_time.count() / queryset._num * 1000
              << "ms" << std::endl;
    std::cout << "query per second: " << 1000.0 / (search_time.count() / queryset._num * 1000) << " qps" << std::endl;
    io.SaveVectorResult(result_path, knns, top_k);
    io.SaveLabelResult(result_path + "_top" + std::to_string(top_k) + ".label", knns_label, top_k);
  }

  return 0;
}