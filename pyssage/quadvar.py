import numpy
from pyssage.classes import Number


def wrap_transect(x: int, n: int) -> int:
    """
    Allows for wrapping an analysis across the ends of a linear transect as if it were a circle

    x is the requested position of the transect
    n is the length of the transect

    assumes zero-delimited indexing (transect positions are counted from 0 to n-1)
    """
    if x >= n:
        return x - n
    elif x < 0:
        return x + n
    else:
        return x


def ttlqv(transect: numpy.array, min_block_size: int = 1, max_block_size: int= 0, block_step: int = 1,
          wrap: bool = False, unit_scale: Number = 1):
    n = len(transect)
    output = []
    if max_block_size == 0:
        max_block_size = n // 2
    if max_block_size < 2:
        max_block_size = 2
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


"""

     // find maximum scale in any data set
     maxb := trunc(maxl * range / 100.0);
     if (maxb > maxl div 2) then maxb := maxl div 2;
     if (maxb < 2) then maxb := 2;

procedure Calc_TTLQV(DatMat,OutMat : TpasMatrix; outcol,maxxb : integer;
          range : double; DoWrap : boolean);
var
   endp,
   cnt,i,j,jj,b,maxb : integer;
   sum1,sum2 : double;
   good : boolean;
begin
     maxb := trunc(DatMat.nrows * range);
     maxb := Min(maxb,maxxb);
     for b := 1 to maxb do if ContinueProgress then begin
         cnt := 0;
         OutMat[b,outcol] := 0.0;
         if DoWrap then endp := DatMat.nrows
         else endp := DatMat.nrows + 1 - 2 * b;
         for i := 1 to endp do if ContinueProgress then begin
             sum1 := 0.0; sum2 := 0.0;
             good := true;
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
             if good then begin
                inc(cnt);
                OutMat[b,outcol] := OutMat[b,outcol] + sqr(sum1 - sum2);
             end;
         end;
         if (cnt > 0) then
            OutMat[b,outcol] := OutMat[b,outcol] / (2.0 * b * cnt)
         else OutMat.IsEmpty[b,outcol] := true;
         ProgressIncrement;
     end;
     if ContinueProgress then
        for i := maxb+1 to maxxb do ProgressIncrement;
end;
"""