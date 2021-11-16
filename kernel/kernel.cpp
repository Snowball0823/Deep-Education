#include <cassert>
#include <iostream>
#include <limits>
#include <thread>
#include <atomic>
#include <vector>
#include <omp.h>

#include "kernel.h"

using std::cout;
using std::endl;

int THD_COUNT = 1;

using std::string;


void _spmm(csr_t* snaph, array2d_t<float> & input, array2d_t<float> & output, 
                op_t op, bool reverse, bool norm /*= true*/)
{   
    // (v_number*v_number) conv (v_number, dim) => v_number*dim
    // if norm, l2_norm(output)
    vid_t* node_index = snaph->offset;
    vid_t* node_value = snaph->nebrs;
    for(int i=0; i<snaph->v_count; i++){
        for(int j=0; j<output.col_count; j++){
            float tmp_conv_num = 0;
            float* output_row_address = output[i];
            tmp_conv_num += (float)i*input[i][j];
            for(int k=node_index[i]; k<node_index[i+1]; k++){
                if(op==eSUM){
                    // cout << "True " << endl;
                    tmp_conv_num += (float)node_value[k]*input[node_value[k]][j];
                }
            }
            output[i][j] = tmp_conv_num;
        }
        if(norm){
            // cout << "Norm" << endl;
            float degree = node_index[i+1] - node_index[i];
            output.row_normalize(i, degree+1);
        }
    }
    // cout << "output col number " << output.col_count << endl;

    // cout << "spmm " << op << "reverse = " << reverse << endl;

    //If in backward, normalize it first, else normalize it after computation
    
    //The core logic goes here.    
}

void _sub_thread_spmm(atomic<int>& row_point, vid_t v_count, vid_t* node_index, vid_t* node_value, 
                        array2d_t<float> & input, array2d_t<float> & output, op_t op, bool reverse, bool norm)
{
    while(1){
        row_point++;
        if(row_point.load()>=v_count){break;}
        for(int j=0; j<output.col_count; j++){
            float tmp_conv_num = 0;
            float* output_row_address = output[row_point];
            tmp_conv_num += (float)row_point*input[row_point][j];
            for(int k=node_index[row_point]; k<node_index[row_point+1]; k++){
                if(op==eSUM){
                    // cout << "True " << endl;
                    tmp_conv_num += (float)node_value[k]*input[node_value[k]][j];
                }
            }
            output[row_point][j] = tmp_conv_num;
            cout<< row_point.load() <<endl;
        }
        if(norm){
            // cout << "Norm" << endl;
            float degree = node_index[row_point+1] - node_index[row_point];
            output.row_normalize(row_point, degree+1);
        }
    }
}

void _multi_thread_spmm_thread(csr_t* snaph, array2d_t<float> & input, array2d_t<float> & output, 
                            op_t op, bool reverse, bool norm /*= true*/, int64_t thread_number)
{
    // (v_number*v_number) conv (v_number, dim) => v_number*dim
    // if norm, l2_norm(output)
    vid_t* node_index = snaph->offset;
    vid_t* node_value = snaph->nebrs;
    atomic<int> row_point(-1);
    vector<thread> thread_list;
    for(int i=0; i<thread_number; i++){
            thread tmp_thread(_sub_thread_spmm, ref(row_point), ref(snaph->v_count), ref(node_index), ref(node_value), ref(input),
                                ref(output), ref(op), ref(reverse), ref(norm));
            thread_list.push_back(move(tmp_thread));
    }
    for(int i=0; i<thread_number; i++){thread_list[i].join();}
    thread_list.clear();
}

void _multi_thread_spmm_opm(csr_t* snaph, array2d_t<float> & input, array2d_t<float> & output, 
                            op_t op, bool reverse, bool norm /*= true*/, int64_t thread_number)
{
    // (v_number*v_number) conv (v_number, dim) => v_number*dim
    // if norm, l2_norm(output)
    vid_t* node_index = snaph->offset;
    vid_t* node_value = snaph->nebrs;
    # pragma omp parallel for num_threads(thread_number)
    for(int i=0; i<snaph->v_count; i++){
        for(int j=0; j<output.col_count; j++){
            float tmp_conv_num = 0;
            float* output_row_address = output[i];
            tmp_conv_num += (float)i*input[i][j];
            for(int k=node_index[i]; k<node_index[i+1]; k++){
                if(op==eSUM){
                    // cout << "True " << endl;
                    tmp_conv_num += (float)node_value[k]*input[node_value[k]][j];
                }
            }
            output[i][j] = tmp_conv_num;
        }
        if(norm){
            // cout << "Norm" << endl;
            float degree = node_index[i+1] - node_index[i];
            output.row_normalize(i, degree+1);
        }
    }    
}

void invoke_spmm(graph_t& graph, array2d_t<float> & input_array, array2d_t<float> & output_array, 
                    bool reverse, bool norm /*= true*/, int64_t thread_number)
{
    if (reverse){
        if (thread_number == 1){
         return _spmm(&graph.csr, input_array, output_array, eSUM, reverse, norm);
        }
        else if (thread_number > 1){
            // return _multi_thread_spmm_thread(&graph.csr, input_array, output_array, eSUM, reverse, norm, thread_number);
             return _multi_thread_spmm_opm(&graph.csr, input_array, output_array, eSUM, reverse, norm, thread_number);
        }
    } 
    else{
         if (thread_number == 1){
         return _spmm(&graph.csc, input_array, output_array, eSUM, reverse, norm);
        }
        else if (thread_number > 1){
            // return _multi_thread_spmm_thread(&graph.csc, input_array, output_array, eSUM, reverse, norm, thread_number);
            return _multi_thread_spmm_opm(&graph.csc, input_array, output_array, eSUM, reverse, norm, thread_number);
        }
    }
}
