from pyssage.utils import check_for_square_matrix
from pyssage.common import OUT_FRMT
from math import sqrt
from typing import Tuple
import numpy
import scipy.stats

__all__ = ["mantel"]


def check_tail(tail: str) -> None:
    """
    checks that the input tail is valid value

    :param tail: the string indicating the desired tale; valid values are "left", "right", and "both"
    """
    valid_tails = ("left", "right", "both")
    if tail not in valid_tails:
        raise ValueError("Requested probability tail is invalid. Options include \"left\", \"right\", or \"both\"")


def mantel(input_matrix1, input_matrix2, partial, permutations: int = 0,
           tail: str = "both") -> Tuple[float, float, list, float, float, float]:
    check_tail(tail)
    n = check_for_square_matrix(input_matrix1)
    if n != check_for_square_matrix(input_matrix2):
        raise ValueError("input matrices must be the same size")

    if len(partial) > 0:
        matrix1 = residuals_from_matrix_regression(input_matrix1, partial)
        matrix2 = residuals_from_matrix_regression(input_matrix2, partial)
    else:
        matrix1 = numpy.copy(input_matrix1)
        matrix2 = numpy.copy(input_matrix2)

    observed_z = numpy.sum(numpy.multiply(matrix1, matrix2))  # assumes diagonals are all zeros

    # for non-partial version the denominator of r is constant no matter the permutation, so only calculate once to
    # save time on two-tailed tests
    sq_cov2 = square_matrix_covariance(matrix2, matrix2)
    sqxy = sqrt(square_matrix_covariance(matrix1, matrix1) * sq_cov2)

    r = square_matrix_covariance(matrix1, matrix2) / sqxy
    observed_mu, observed_std = mantel_moments(matrix1, matrix2)
    z_score = (observed_z - observed_mu) / observed_std
    p_value = scipy.stats.norm.cdf(z_score)

    # create basic output text
    output_text = list()
    output_text.append("Mantel Test")
    output_text.append("")
    # matrix information here??
    # OutputAddLine('Matrix 1: ' + iMat1.MatrixName);
    # OutputAddLine('Matrix 2: ' + iMat2.MatrixName);
    # if (MList.Count > 0) then begin
    #    outstr := 'Matrices held constant: ';
    #    for i := 0 to MList.Count - 1 do begin
    #        if (i > 0) then outstr := outstr + ', ';
    #        outstr := outstr + TpasBasicMatrix(MList[i]).MatrixName;
    #    end;
    #    OutputAddLine(outstr);
    # end;
    output_text.append("Matrices are {0} x {0}".format(n))
    print()
    output_text.append("Observed Z = " + format(observed_z, OUT_FRMT))
    output_text.append("Correlation = " + format(r, OUT_FRMT))
    output_text.append("t = " + format(z_score, OUT_FRMT))
    output_text.append("Left-tailed p = " + format(p_value, OUT_FRMT))
    output_text.append("Right-tailed p = " + format(1 - p_value, OUT_FRMT))
    output_text.append("Two-tailed p = " + format(2*min(p_value, 1 - p_value), OUT_FRMT))
    output_text.append("")

    # change p_value to requested tail
    if tail == "both":
        p_value = 2*min(p_value, 1 - p_value)
    elif tail == "right":
        p_value = 1 - p_value

    # perform permutation tests
    if permutations > 0:
        cumulative_left = 0
        cumulative_right = 0
        cumulative_equal = 1
        cumulative_total = 1
        for p in range(permutations):
            # matrix1 = permute_matrix(input_matrix1)

            if len(partial) > 0:  # if partial, calculate residuals for permuted matrix
                matrix1 = residuals_from_matrix_regression(matrix1, partial)
            # If it is a two-tailed test, we need to calculate r, otherwise for one-tailed tests we can stick with
            # Z which is faster
            if tail == "both":
                numerator = square_matrix_covariance(matrix1, matrix2)
                if len(partial > 0):
                    denominator = sqrt(square_matrix_covariance(matrix1, matrix1) * sq_cov2)
                else:  # for non-partial tests can save computation as denominator is fixed
                    denominator = sqxy
                permuted_r = numerator / denominator
                if permuted_r < r:
                    cumulative_left += 1
                elif permuted_r > r:
                    cumulative_right += 1
                else:
                    cumulative_equal += 1
                if abs(permuted_r) >= abs(r):
                    cumulative_total += 1
            else:
                permuted_z = numpy.sum(numpy.multiply(matrix1, matrix2))
                if permuted_z < observed_z:
                    cumulative_left += 1
                elif permuted_z > observed_z:
                    cumulative_right += 1
                else:
                    cumulative_equal += 1
        permuted_right_p = (cumulative_equal + cumulative_right) / (permutations + 1)
        permuted_left_p = (cumulative_equal + cumulative_left) / (permutations + 1)
        if tail == "both":
            permuted_two_p = cumulative_total / (permutations + 1)
        else:
            permuted_two_p = 1
        output_text.append("Probability results from {} permutation".format(permutations))
        output_text.append("# of permutations < observed = {}".format(cumulative_left))
        output_text.append("# of permutations > observed = {}".format(cumulative_right))
        if tail == "both":
            output_text.append("# of permutations >= |observed| = {}".format(cumulative_total))
        output_text.append("# of permutations = observed = {}".format(cumulative_equal))
        output_text.append("")
        output_text.append("Left-tailed p = " + format(permuted_left_p, OUT_FRMT))
        output_text.append("Right-tailed p = observed = " + format(permuted_right_p, OUT_FRMT))
        if tail == "both":
            output_text.append("Two-tailed p = observed = " + format(permuted_two_p, OUT_FRMT))
        output_text.append("")
    else:
        permuted_left_p, permuted_right_p, permuted_two_p = 1, 1, 1

    return r, p_value, output_text, permuted_left_p, permuted_right_p, permuted_two_p


def mantel_moments(x: numpy.ndarray, y: numpy.ndarray) -> Tuple[float, float]:
    """
    calculates the moments for a Mantel test

    assumes diagonal elements bot both input matrices are zeros

    :param x: first matrix, as a square numpy.ndarray
    :param y: second matrix, as a square numpy.ndarray
    :return: expected value (mu) and standard deviation as a tuple
    """
    n = len(x)
    ax, bx, cx, dx, ex, fx, gx, hx, ix, jx, kx = matrix_sums(x)
    ay, by, cy, dy, ey, fy, gy, hy, iy, jy, ky = matrix_sums(y)
    mu = ax*ay / (n*(n-1))
    obsvar = bx*by + cx*cy + (hx*hy + ix*iy + 2*jx*jy)/(n-2) + kx*ky/(n-2)/(n-3) - gx*gy/n/(n-1)
    obsvar /= n*(n - 1)
    return mu, sqrt(obsvar)


def matrix_sums(x: numpy.ndarray):
    """
    calculates a bunch of sums and sums of squares for different elements of a square matrix

    assumes diagonal elements are zero
    """
    aa = numpy.sum(x)  # sum of all values in matrix
    bb = numpy.sum(numpy.square(x))  # sum of all squared values in matrix
    cc = numpy.sum(x * numpy.transpose(x))  # sum of all corresponding elements; same as bb if matrix is symmetric
    row_sums = numpy.sum(x, axis=1)  # vector containing sum of each row
    col_sums = numpy.sum(x, axis=0)  # vector containing sum of each column
    dd = numpy.sum(numpy.square(col_sums))  # sum of squared column sums
    ee = numpy.sum(numpy.square(row_sums))  # sum of squared row sums
    ff = numpy.sum(numpy.multiply(row_sums, col_sums))  # sum of product of corresponding row and columns sums
    gg = aa**2
    hh = dd - bb
    ii = ee - bb
    jj = ff - cc
    kk = gg + bb + cc - dd - ee - 2*ff
    return aa, bb, cc, dd, ee, ff, gg, hh, ii, jj, kk


def residuals_from_matrix_regression(y_matrix: numpy.ndarray, x: list) -> numpy.ndarray:
    if len(x) == 1:
        return residuals_from_simple_matrix_regression(y_matrix, x[0])
    else:
        return residuals_from_multi_matrix_regression(y_matrix, x)


def residuals_from_simple_matrix_regression(y: numpy.ndarray, x: numpy.ndarray) -> numpy.ndarray:
    """
    performs a linear regression of matrix y on matrix x and returns the residuals

    assumes the diagonals of the input matrices are all zeros

    :param y: matrix of dependent variables, as a square numpy.ndarray
    :param x: matrix of independenet variables, as a square numpy.ndarray
    :return: matrix of residuals of the simple linear regression of y on x
    """
    n = check_for_square_matrix(y)
    if n != check_for_square_matrix(x):
        raise ValueError("matrices must be the same size")

    sumx = numpy.sum(x)
    sumy = numpy.sum(y)
    sumx2 = numpy.sum(numpy.square(x))
    sumxy = numpy.sum(numpy.multiply(x, y))
    count = n**2 - n  # number of off-diagonal elements
    xbar = sumx / count
    ybar = sumy / count
    beta = (sumxy - count*xbar*ybar) / (sumx2 - count*xbar**2)
    alpha = ybar - beta*xbar

    residuals = y - alpha - beta*x
    numpy.fill_diagonal(residuals, 0)  # reset diagonal elements to zero

    return residuals


def residuals_from_multi_matrix_regression(y: numpy.ndarray, x_list: list) -> numpy.ndarray:
    """
    performs a muliple linear regression of matrix y on all of the matrices in x and returns the residuals
    """
    n = check_for_square_matrix(y)
    for x in x_list:
        if n != check_for_square_matrix(x):
            raise ValueError("matrices must be the same size")

    """
         // count number of elements for regression
         cnt := 0;
         for i := 1 to n do
             for j := 1 to n do
                 if (i <> j) then
                    if GoodMat[i,j] then inc(cnt);
         // construct input data arrays
         SetLength(Yarray,2,cnt+1);
         SetLength(Xarray,XList.Count+2,cnt+1);
         // set first column to all 1's
         // store zero's
    
         for i := 1 to cnt do Xarray[1,i] := 1.0;
         k := 0;
         for i := 1 to n do
             for j := 1 to n do
                 if (i <> j) then
                    if GoodMat[i,j] then begin
                       inc(k);
                       case yt of
                            1 : Yarray[1,k] := TpasSymmetricMatrix(YMat)[order[i],order[j]];
                            2 : if TpasBooleanMatrix(YMat)[order[i],order[j]] then
                                   Yarray[1,k] := 1.0
                                else Yarray[1,k] := 0.0;
                            3 : Yarray[1,k] := TpasAngleMatrix(YMat)[order[i],order[j]];
                       end;
                       for m := 0 to XList.Count - 1 do begin
                           tMat := TpasBasicMatrix(XList[m]);
                           case xt[m] of
                                1 : Xarray[m+2,k] := TpasSymmetricMatrix(tMat)[i,j];
                                2 : if TpasBooleanMatrix(tMat)[i,j] then
                                       Xarray[m+2,k] := 1.0
                                    else Xarray[m+2,k] := 0.0;
                                3 : Xarray[m+2,k] := TpasAngleMatrix(tMat)[i,j];
                           end;
                       end;
                    end;
         // do multiple regression
    
         //tmpoutput(xarray,'xarray.txt');
         //tmpoutput(yarray,'yarray.txt');
    
         Xtrans := MatrixTransposition(Xarray);
    
         //tmpoutput(xtrans,'xtrans.txt');
    
         XtX := MatrixMultiply(Xtrans,Xarray);
    
         //tmpoutput(xtx,'xtx.txt');
    
         InvXtX := MatrixInverse(XtX);
    
         //tmpoutput(Invxtx,'InvXtX.txt');
    
         XtXIXt := MatrixMultiply(InvXtX,Xtrans);
    
         //tmpoutput(xtxixt,'xtxixt.txt');
    
         Barray := MatrixMultiply(XtXIXt,Yarray);
    
         //tmpoutput(barray,'barray.txt');
    
         Yhat := MatrixMultiply(Xarray,Barray);
    
         //tmpoutput(yhat,'yhat.txt');
    
         // output residuals
         ResMat := TpasMatrix.Create(n,n);
         k := 0;
         for i := 1 to n do
             for j := 1 to n do
                 if (i <> j) then
                    if GoodMat[i,j] then begin
                       inc(k);
                       ResMat[i,j] := Yarray[1,k] - Yhat[1,k];
                    end;
    """


def square_matrix_covariance(x: numpy.ndarray, y: numpy.ndarray) -> float:
    """
    returns the covariance of two square matrices, assuming the diagonals are both zeros

    :param x: first matrix, as a square numpy.ndarray
    :param y: second matrix, as a square numpy.ndarray
    :return: the covariance of the two matrices
    """
    n = check_for_square_matrix(y)
    if n != check_for_square_matrix(x):
        raise ValueError("matrices must be the same size")

    sumx = numpy.sum(x)
    sumy = numpy.sum(y)
    sumxy = numpy.sum(numpy.multiply(x, y))
    count = n**2 - n

    return (sumxy - sumx*sumy/count) / (count - 1)
