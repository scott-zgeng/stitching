// mat_proc.cpp write by scott.zgeng 2015.10.26


#include "mat_proc.h"


extern "C" {
#include "meschach/matrix.h"
#include "meschach/matrix2.h"
}


// 只需要支持 MV_64FC1
mv_mat_handle mv_create_matrix(int rows, int cols)
{
    return m_get(rows, cols);
}


void mv_release_matrix(mv_mat_handle mat)
{
    m_free((MAT*)mat);
}


mv_mat_handle mv_clone_matrix(const mv_mat_handle mat)
{
    MAT* src = (MAT*)mat;
    MAT* dst = m_get(src->m, src->n);
    return m_copy(src, dst);    
}


void mv_matrix_zero(mv_mat_handle mat)
{
    m_zero((MAT*)mat);
}


void mv_matrix_set(mv_mat_handle mat, int row, int col, double value)
{
    m_set_val((MAT*)mat, row, col, value);
}


bool mv_invert(mv_mat_handle src, mv_mat_handle dst)
{
    MAT* out = m_inverse((MAT*)src, (MAT*)dst);
    return (out != NULL);
}


double mv_invert_svd(mv_mat_handle src, mv_mat_handle dst)
{
    //m_inverse((MAT*)src, (MAT*)dst);
    // 该函数需要实现
    return 0.0;
}


void mv_matrix_mul(const mv_mat_handle src1, const mv_mat_handle src2, mv_mat_handle dst)
{
    m_mlt((MAT*)src1, (MAT*)src2, (MAT*)dst);

    //mv_mltadd
}


double mv_matrix_get(mv_mat_handle mat, int row, int col)
{
    return m_get_val((MAT*)mat, row, col);
}


int mv_solve(const mv_mat_handle src1, const mv_mat_handle src2, mv_vec_handle dst)
{   
    svd((MAT*)src1, (MAT*)src2, NULL, (VEC*)dst);
    return 0;
}


mv_vec_handle mv_create_vector(int n)
{
    return v_get(n);
}

void mv_release_vector(mv_vec_handle vec)
{
    v_free((VEC*)vec);
}

double mv_vector_get(mv_vec_handle vec, int n)
{
    return v_get_val((VEC*)vec, n);
}
