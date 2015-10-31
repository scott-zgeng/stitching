// mat_proc.h write by scott.zgeng 2015.10.26


#ifndef  __MAT_PROC_H__
#define  __MAT_PROC_H__


// todo(scott.zgeng): 目前先简单封装MAT，后续需要完善时再改
typedef void* mv_mat_handle;
typedef void* mv_vec_handle;

mv_mat_handle mv_create_matrix(int rows, int cols);
void mv_release_matrix(mv_mat_handle mat);
mv_mat_handle mv_clone_matrix(const mv_mat_handle mat);


void mv_matrix_zero(mv_mat_handle mat);
void mv_matrix_set(mv_mat_handle mat, int row, int col, double value);


//CV_LU
// Gaussian elimination with optimal pivot element chose In case of LU method 
// the function returns src1 determinant(src1 must be square).If it is 0, the
// matrix is not inverted and src2 is filled with zeros.
bool mv_invert(mv_mat_handle src, mv_mat_handle dst);

//CV_SVD
// Singular value decomposition(SVD) method In case of SVD methods the function
// returns the inversed condition number of src1(ratio of the smallest singular 
// value to the largest singular value) and 0 if src1 is all zeros. The SVD  
// methods calculate a pseudo - inverse matrix if src1 is singular
double mv_invert_svd(mv_mat_handle src, mv_mat_handle dst);

void mv_matrix_mul(const mv_mat_handle src1, const mv_mat_handle src2, mv_mat_handle dst);


double mv_matrix_get(mv_mat_handle mat, int row, int col);

int mv_solve(const mv_mat_handle src1, const mv_mat_handle src2, mv_vec_handle dst);   // SVD 

mv_vec_handle mv_create_vector(int n);
void mv_release_vector(mv_vec_handle vec);
double mv_vector_get(mv_vec_handle vec, int n);

#endif //__MAT_PROC_H__

