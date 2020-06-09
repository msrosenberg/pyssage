import numpy
from pyssage.classes import Number

__all__ = ["ttlqv", "three_tlqv", "pqv"]


def wrap_transect(x: int, n: int) -> int:
    """
    Allows for wrapping an analysis across the ends of a linear transect as if it were a circle.
    Assumes zero-delimited indexing (transect positions are counted from 0 to n-1)

    :param x: the index of the requested position of the transect
    :param n: the length of the transect
    :return: returns the index of the correct position within the transect
    """
    if x >= n:
        return x - n
    elif x < 0:
        return x + n
    else:
        return x


def check_block_size(max_block_size: int, n: int, x: int) -> int:
    if max_block_size == 0:
        max_block_size = n // x
    if max_block_size < 2:
        max_block_size = 2
    elif max_block_size > n // x:
        max_block_size = n // x
        print("Maximum block size cannot exceed {:0.0f}% of transect length. Reduced to {}.".format(100 / x,
                                                                                                    max_block_size))
    return max_block_size


def ttlqv(transect: numpy.ndarray, min_block_size: int = 1, max_block_size: int = 0, block_step: int = 1,
          wrap: bool = False, unit_scale: Number = 1) -> numpy.ndarray:
    """
    Performs a Two-Term Local Quadrat Variance analysis (TTLQV) on a transect. Method originally from:

    Hill, M.O. 1973. The intensity of spatial pattern in plant communities. Journal of Ecology 61:225-235.

    :param transect: a single dimensional numpy array containing the transect data
    :param min_block_size: the smallest block size of the analysis (default = 1)
    :param max_block_size: the largest block size of the analysis (default = 0, indicating 50% of the transect length)
    :param block_step: the incremental size increase of each block size (default = 1)
    :param wrap: treat the transect as a circle where the ends meet (default = False)
    :param unit_scale: represents the unit scale of a single block (default = 1). Can be used to rescale the units of
           the output, e.g., if the blocks are measured in centimeters, you could use a scale of 0.01 to have the
           output expressed in meters.
    :return: a two column numpy array, with the first column containing the scaled block size and the second the
             calculated variance
    """
    n = len(transect)
    output = []
    max_block_size = check_block_size(max_block_size, n, 2)
    for b in range(min_block_size, max_block_size + 1, block_step):
        cnt = 0
        qv = 0
        if wrap:
            end_start_pos = n
        else:
            end_start_pos = n + 1 - 2*b
        for start_pos in range(end_start_pos):
            sum1 = 0
            sum2 = 0
            for i in range(start_pos, start_pos + b):
                j = wrap_transect(i, n)
                sum1 += transect[j]
            for i in range(start_pos + b, start_pos + 2*b):
                j = wrap_transect(i, n)
                sum2 += transect[j]
            cnt += 1
            qv += (sum1 - sum2)**2
        qv /= 2*b*cnt
        output.append([b*unit_scale, qv])
    return numpy.array(output)


def three_tlqv(transect: numpy.ndarray, min_block_size: int = 1, max_block_size: int = 0, block_step: int = 1,
               wrap: bool = False, unit_scale: Number = 1) -> numpy.ndarray:
    """
    Performs a Three-Term Local Quadrat Variance analysis (3TLQV) on a transect. Method originally from:

    Hill, M.O. 1973. The intensity of spatial pattern in plant communities. Journal of Ecology 61:225-235.

    :param transect: a single dimensional numpy array containing the transect data
    :param min_block_size: the smallest block size of the analysis (default = 1)
    :param max_block_size: the largest block size of the analysis (default = 0, indicating 33% of the transect length)
    :param block_step: the incremental size increase of each block size (default = 1)
    :param wrap: treat the transect as a circle where the ends meet (default = False)
    :param unit_scale: represents the unit scale of a single block (default = 1). Can be used to rescale the units of
           the output, e.g., if the blocks are measured in centimeters, you could use a scale of 0.01 to have the
           output expressed in meters.
    :return: a two column numpy array, with the first column containing the scaled block size and the second the
             calculated variance
    """
    n = len(transect)
    output = []
    max_block_size = check_block_size(max_block_size, n, 3)
    for b in range(min_block_size, max_block_size + 1, block_step):
        cnt = 0
        qv = 0
        if wrap:
            end_start_pos = n
        else:
            end_start_pos = n + 1 - 3*b
        for start_pos in range(end_start_pos):
            sum1 = 0
            sum2 = 0
            sum3 = 0
            for i in range(start_pos, start_pos + b):
                j = wrap_transect(i, n)
                sum1 += transect[j]
            for i in range(start_pos + b, start_pos + 2*b):
                j = wrap_transect(i, n)
                sum2 += transect[j]
            for i in range(start_pos + 2*b, start_pos + 3*b):
                j = wrap_transect(i, n)
                sum3 += transect[j]
            cnt += 1
            qv += (sum1 - 2*sum2 + sum3)**2
        qv /= 8*b*cnt
        output.append([b * unit_scale, qv])
    return numpy.array(output)


def pqv(transect: numpy.ndarray, min_block_size: int = 1, max_block_size: int = 0, block_step: int = 1,
        wrap: bool = False, unit_scale: Number = 1) -> numpy.ndarray:
    """
    Performs a Paired Quadrat Variance analysis (PQV) on a transect.

    :param transect: a single dimensional numpy array containing the transect data
    :param min_block_size: the smallest block size of the analysis (default = 1)
    :param max_block_size: the largest block size of the analysis (default = 0, indicating 50% of the transect length)
    :param block_step: the incremental size increase of each block size (default = 1)
    :param wrap: treat the transect as a circle where the ends meet (default = False)
    :param unit_scale: represents the unit scale of a single block (default = 1). Can be used to rescale the units of
           the output, e.g., if the blocks are measured in centimeters, you could use a scale of 0.01 to have the
           output expressed in meters.
    :return: a two column numpy array, with the first column containing the scaled block size and the second the
             calculated variance
    """
    n = len(transect)
    output = []
    max_block_size = check_block_size(max_block_size, n, 2)
    for b in range(min_block_size, max_block_size + 1, block_step):
        cnt = 0
        qv = 0
        if wrap:
            end_start_pos = n
        else:
            end_start_pos = n - b
        for start_pos in range(end_start_pos):
            for i in range(end_start_pos):
                j = wrap_transect(i + b, n)
                cnt += 1
                qv += (transect[i] - transect[j])**2
        qv /= 2*cnt
        output.append([b*unit_scale, qv])
    return numpy.array(output)


def tqv(transect: numpy.ndarray, min_block_size: int = 1, max_block_size: int = 0, block_step: int = 1,
        wrap: bool = False, unit_scale: Number = 1) -> numpy.ndarray:
    """
    Performs a Triplet Quadrat Variance analysis (tQV) on a transect. Method originally from:

    Dale, M.R T. 1999. Spatial Pattern Analysis in Plant Ecology. Cambridge: Cambridge University Press.

    :param transect: a single dimensional numpy array containing the transect data
    :param min_block_size: the smallest block size of the analysis (default = 1)
    :param max_block_size: the largest block size of the analysis (default = 0, indicating 50% of the transect length)
    :param block_step: the incremental size increase of each block size (default = 1)
    :param wrap: treat the transect as a circle where the ends meet (default = False)
    :param unit_scale: represents the unit scale of a single block (default = 1). Can be used to rescale the units of
           the output, e.g., if the blocks are measured in centimeters, you could use a scale of 0.01 to have the
           output expressed in meters.
    :return: a two column numpy array, with the first column containing the scaled block size and the second the
             calculated variance
    """
    n = len(transect)
    output = []
    max_block_size = check_block_size(max_block_size, n, 3)
    for b in range(min_block_size, max_block_size + 1, block_step):
        cnt = 0
        qv = 0
        if wrap:
            end_start_pos = n
        else:
            end_start_pos = n - 2*b
        for start_pos in range(end_start_pos):
            for i in range(end_start_pos):
                j = wrap_transect(i + b, n)
                k = wrap_transect(i + 2*b, n)
                cnt += 1
                qv += (transect[i] - 2*transect[j] + transect[k])**2
        qv /= 4*cnt
        output.append([b*unit_scale, qv])
    return numpy.array(output)


def two_nlv(transect: numpy.ndarray, min_block_size: int = 1, max_block_size: int = 0, block_step: int = 1,
            wrap: bool = False, unit_scale: Number = 1) -> numpy.ndarray:
    """
    Performs a Two-Term New Local Variance analysis (2NLV) on a transect. Method originally from:

    Galiano, E.F. 1982. Détection et mesure de l'hétérogénéité spatiale des espèces dans les pâturages.
    Acta OEcologia / OEcologia Plantarum 3:269-278.

    :param transect: a single dimensional numpy array containing the transect data
    :param min_block_size: the smallest block size of the analysis (default = 1)
    :param max_block_size: the largest block size of the analysis (default = 0, indicating 50% of the transect length)
    :param block_step: the incremental size increase of each block size (default = 1)
    :param wrap: treat the transect as a circle where the ends meet (default = False)
    :param unit_scale: represents the unit scale of a single block (default = 1). Can be used to rescale the units of
           the output, e.g., if the blocks are measured in centimeters, you could use a scale of 0.01 to have the
           output expressed in meters.
    :return: a two column numpy array, with the first column containing the scaled block size and the second the
             calculated variance
    """
    n = len(transect)
    output = []
    max_block_size = check_block_size(max_block_size, n, 2)
    for b in range(min_block_size, max_block_size + 1, block_step):
        cnt = 0
        qv = 0
        if wrap:
            end_start_pos = n
        else:
            end_start_pos = n - 2*b
        for start_pos in range(end_start_pos):
            sum1 = 0
            sum2 = 0
            for i in range(start_pos, start_pos + b):
                j = wrap_transect(i, n)
                sum1 += transect[j]
            for i in range(start_pos + b, start_pos + 2*b):
                j = wrap_transect(i, n)
                sum2 += transect[j]
            term1 = (sum1 - sum2)**2
            sum1 = 0
            sum2 = 0
            for i in range(start_pos + 1, start_pos + b + 1):
                j = wrap_transect(i, n)
                sum1 += transect[j]
            for i in range(start_pos + b + 1, start_pos + 2*b + 1):
                j = wrap_transect(i, n)
                sum2 += transect[j]
            term2 = (sum1 - sum2)**2
            cnt += 1
            qv += abs(term1 - term2)
        if cnt > 0:
            qv /= 2*b*cnt
            output.append([b*unit_scale, qv])
    return numpy.array(output)


def three_nlv(transect: numpy.ndarray, min_block_size: int = 1, max_block_size: int = 0, block_step: int = 1,
              wrap: bool = False, unit_scale: Number = 1) -> numpy.ndarray:
    """
    Performs a Three-Term New Local Variance analysis (2NLV) on a transect. Method originally from:

    Galiano, E.F. 1982. Détection et mesure de l'hétérogénéité spatiale des espèces dans les pâturages.
    Acta OEcologia / OEcologia Plantarum 3:269-278.

    :param transect: a single dimensional numpy array containing the transect data
    :param min_block_size: the smallest block size of the analysis (default = 1)
    :param max_block_size: the largest block size of the analysis (default = 0, indicating 50% of the transect length)
    :param block_step: the incremental size increase of each block size (default = 1)
    :param wrap: treat the transect as a circle where the ends meet (default = False)
    :param unit_scale: represents the unit scale of a single block (default = 1). Can be used to rescale the units of
           the output, e.g., if the blocks are measured in centimeters, you could use a scale of 0.01 to have the
           output expressed in meters.
    :return: a two column numpy array, with the first column containing the scaled block size and the second the
             calculated variance
    """
    n = len(transect)
    output = []
    max_block_size = check_block_size(max_block_size, n, 3)
    for b in range(min_block_size, max_block_size + 1, block_step):
        cnt = 0
        qv = 0
        if wrap:
            end_start_pos = n
        else:
            end_start_pos = n - 3*b
        for start_pos in range(end_start_pos):
            sum1 = 0
            sum2 = 0
            sum3 = 0
            for i in range(start_pos, start_pos + b):
                j = wrap_transect(i, n)
                sum1 += transect[j]
            for i in range(start_pos + b, start_pos + 2*b):
                j = wrap_transect(i, n)
                sum2 += transect[j]
            for i in range(start_pos + 2*b, start_pos + 3*b):
                j = wrap_transect(i, n)
                sum3 += transect[j]
            term1 = (sum1 - 2*sum2 + sum3)**2
            sum1 = 0
            sum2 = 0
            sum3 = 0
            for i in range(start_pos + 1, start_pos + b + 1):
                j = wrap_transect(i, n)
                sum1 += transect[j]
            for i in range(start_pos + b + 1, start_pos + 2*b + 1):
                j = wrap_transect(i, n)
                sum2 += transect[j]
            for i in range(start_pos + 2*b + 1, start_pos + 3*b + 1):
                j = wrap_transect(i, n)
                sum3 += transect[j]
            term2 = (sum1 - 2*sum2 + sum3)**2
            cnt += 1
            qv += abs(term1 - term2)
        if cnt > 0:
            qv /= 8*b*cnt
            output.append([b*unit_scale, qv])
    return numpy.array(output)

"""
procedure Calc_3NLV(DatMat,OutMat : TpasMatrix; outcol,maxxb : integer;
          range : double; DoWrap : boolean);
var
   jj,endp,
   cnt,i,j,b,maxb : integer;
   term1,term2,sum1,sum2,sum3 : double;
   good : boolean;
begin
     maxb := trunc(DatMat.nrows * range);
     maxb := Min(maxb,maxxb);
     for b := 1 to maxb do if ContinueProgress then begin
         cnt := 0;
         OutMat[b,outcol] := 0.0;
         if DoWrap then endp := DatMat.nrows
         else endp := DatMat.nrows - 3 * b;
         for i := 1 to endp do if ContinueProgress then begin
             good := true;
             sum1 := 0.0; sum2 := 0.0; sum3 := 0.0;
             for j := i to i + b - 1 do begin
                 jj := WrapTransect(j,DatMat.nrows);
                 if DatMat.IsNum[jj,1] then
                    sum1 := sum1 + DatMat[jj,1]
                 else good := false;
             end;
             for j := i + b to i + 2 * b - 1 do begin
                 jj := WrapTransect(j,DatMat.nrows);
                 if DatMat.IsNum[jj,1] then
                    sum2 := sum2 + DatMat[jj,1]
                 else good := false;
             end;
             for j := i + 2 * b to i + 3 * b - 1 do begin
                 jj := WrapTransect(j,DatMat.nrows);
                 if DatMat.IsNum[jj,1] then
                    sum3 := sum3 + DatMat[jj,1]
                 else good := false;
             end;
             term1 := sqr(sum1 - 2.0 * sum2 + sum3);
             sum1 := 0.0; sum2 := 0.0; sum3 := 0.0;
             for j := i + 1 to i + b do begin
                 jj := WrapTransect(j,DatMat.nrows);
                 if DatMat.IsNum[jj,1] then
                    sum1 := sum1 + DatMat[jj,1]
                 else good := false;
             end;
             for j := i + b + 1 to i + 2 * b do begin
                 jj := WrapTransect(j,DatMat.nrows);
                 if DatMat.IsNum[jj,1] then
                    sum2 := sum2 + DatMat[jj,1]
                 else good := false;
             end;
             for j := i + 2 * b + 1 to i + 3 * b do begin
                 jj := WrapTransect(j,DatMat.nrows);
                 if DatMat.IsNum[jj,1] then
                    sum3 := sum3 + DatMat[jj,1]
                 else good := false;
             end;
             term2 := sqr(sum1 - 2.0 * sum2 + sum3);
             if good then begin
                inc(cnt);
                OutMat[b,outcol] := OutMat[b,outcol] + abs(term1 - term2);
             end;
         end;
         if (cnt > 0) then
            OutMat[b,outcol] := OutMat[b,outcol] / (8.0 * b * cnt)
         else OutMat.IsEmpty[b,outcol] := true;
         ProgressIncrement;
     end;
     if ContinueProgress then
        for i := maxb+1 to maxxb do ProgressIncrement;
end;

"""



"""

procedure Calc_rPQV(DatMat,OutMat : TpasMatrix; outcol,maxxb : integer;
          range : double; DoWrap : boolean);
var
   maxr,rcnt,cnt,i,j,k,b,maxb : integer;
   Used : TpasBooleanArray;
   Cnts : TpasIntegerArray;
begin
     maxb := trunc(DatMat.nrows * range);
     maxb := Min(maxb,maxxb);
     SetLength(Used,DatMat.nrows+1);
     SetLength(Cnts,maxb);
     for b := 1 to maxb do begin
         OutMat[b,outcol] := 0.0;
         Cnts[b-1] := 0;
     end;
     cnt := 0;
     Used[0] := true;
     for i := 1 to DatMat.nrows do
         if DatMat.IsNum[i,1] then begin
            Used[i] := false;
            inc(cnt);
         end else Used[i] := true;
     j := 1;
     maxr := DatMat.nrows;
     for i := 1 to cnt div 2 do begin
         while Used[j] do inc(j);
         rcnt := 0;
         repeat
               inc(rcnt);
               k := rand(seed,j+1,DatMat.nrows);
               b := k - j;
               if DoWrap then b := Min(b,j+DatMat.nrows-k);
         until (not Used[k] and (b <= maxb)) or (rcnt = maxr);
         if (rcnt <> maxr) then begin
            OutMat[b,outcol] := OutMat[b,outcol] + sqr(DatMat[j,1] - DatMat[k,1]) / 2.0;
            inc(Cnts[b-1]);
            Used[k] := true;
         end;
         Used[j] := true;
     end;
     for b := 1 to maxb do if ContinueProgress then begin
         if (Cnts[b-1] > 0) then OutMat[b,outcol] := OutMat[b,outcol] / Cnts[b-1]
         else OutMat.IsEmpty[b,outcol] := true;
         ProgressIncrement;
     end;
     Used := nil;
     Cnts := nil;
     if ContinueProgress then
        for i := maxb+1 to maxxb do ProgressIncrement;
end;




"""