from typing import Tuple
from collections import namedtuple
from math import sin, cos, pi, sqrt, exp, log
import numpy
from pyssage.classes import Number
from pyssage.utils import check_block_size


#  ---------------Wavelet Functions---------------
def haar_wavelet(d: float) -> int:
    if -1 <= d < 0:
        return -1
    elif 0 <= d < 1:
        return 1
    else:
        return 0


def french_tophat_wavelet(d: float) -> int:
    if 0.5 <= abs(d) < 1.5:
        return -1
    elif abs(d) < 0.5:
        return 2
    else:
        return 0


def mexican_hat_wavelet(d: float) -> float:
    return (2/sqrt(3)) * pi**(-1/4) * (1 - 4*d**2) * exp(-2*d**2)


def morlet_wavelet(d: float) -> float:
    # returns the real part of the Morlet wavelet
    return pi**(-1/4) * cos((d/2)*pi*sqrt(2/log(2))) * exp(-1 * d**2 / 8)


def sine_wavelet(d: float) -> float:
    if abs(d) > 1:
        return 0
    else:
        return sin(pi*d)


# ---------------Primary Function---------------
def wavelet_analysis(transect: numpy.ndarray, wavelet=haar_wavelet, min_block_size: int = 1, max_block_size: int = 0,
                     block_step: int = 1, wrap: bool = False, unit_scale: Number = 1) -> Tuple[numpy.ndarray,
                                                                                               numpy.ndarray,
                                                                                               numpy.ndarray]:
    """
    Performs a wavelet analysis on a transect

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
    # output = []
    max_block_size = check_block_size(max_block_size, n, 2)
    if wrap:
        _transect = numpy.append(transect, transect)
        _transect = numpy.append(transect, _transect)
    else:
        _transect = transect
    start_start_pos = n
    end_start_pos = 2*n
    win_width = 0.5
    if wavelet == french_tophat_wavelet:
        win_width = 1.5
    elif wavelet == mexican_hat_wavelet:
        win_width = 2
    elif wavelet == morlet_wavelet:
        win_width = 6

    v_output = []
    # w_output = {}
    w_output = []
    for b in range(min_block_size, max_block_size + 1, block_step):
        # for p in range(n):  # set default value for w_output matrix
        #     w_output[b, p] = None
        w_row = [numpy.NaN for p in range(n)]
        w_output.append(w_row)

        v = 0
        if not wrap:
            start_start_pos = round(win_width*b)
            end_start_pos = round(n + 1 - win_width*b)
        cnt = 0
        for start_pos in range(start_start_pos, end_start_pos):
            if not wrap:
                actual_p = start_pos
            else:
                actual_p = start_pos - n
            startq = start_pos - round(win_width * b)
            endq = start_pos + round(win_width * b)
            if not wrap:
                startq = max(startq, 0)
                endq = min(endq, n)
            tmp_sum = 0
            for i in range(startq, endq):
                d = (i - start_pos) / b
                tmp_sum += _transect[i] * wavelet(d)
            # w_output[b, actual_p] = tmp_sum/b
            w_row[actual_p] = tmp_sum/b

            v += (tmp_sum/b)**2
            cnt += 1
        if cnt > 0:
            v_output.append([b*unit_scale, v/cnt])
    p_output = []
    for p in range(n):
        pv = 0
        cnt = 0
        # for b in range(min_block_size, max_block_size + 1, block_step):
        #     if (b, p) in w_output:
        #         pv += w_output[b, p]**2
        #         cnt += 1
        for b in range(len(w_output)):
            if w_output[b][p] != numpy.NaN:
                pv += w_output[b][p]**2
                cnt += 1

        if cnt > 0:
            p_output.append([p*unit_scale, pv / cnt])

    wavelet_output_tuple = namedtuple("wavelet_output_tuple", ["w_output", "v_output", "p_output"])
    return wavelet_output_tuple(numpy.array(w_output), numpy.array(v_output), numpy.array(p_output))


"""
{----------Wavelet Analysis----------}

procedure Calc_1DWavelet(Wavelet : integer; DatMat,VOutMat,POutMat,WOutMat : TpasMatrix;
          outcol, maxxb : integer; range : double; DoWrap : boolean);
var
   winwidth : double;
   //winwidth,
   startq,endq,
   cnt,i,x,jj,b,maxb : integer;
   tmpm : double;
   good : boolean;
begin
     maxb := trunc(DatMat.nrows * range);
     maxb := Min(maxb,maxxb);
     // Set frame bounds to reduce computation time
     case Wavelet of
          1 : winwidth := 0.5; // haar
          2 : winwidth := 1.5; // french top hat
          3 : winwidth := 2.0; // mexhat
          4 : winwidth := 2.0; // morlet
          5 : winwidth := 0.5; // sine
          else winwidth := 0.5;
     end;

     for b := 1 to maxb do if ContinueProgress then begin
         VOutMat[b,outcol] := 0.0;
         cnt := 0;
         for x := 1 to DatMat.nrows do if ContinueProgress then begin
             good := true;
             startq := x - round(winwidth * b);
             endq := x + round(winwidth * b);
             if DoWrap or ((startq > 0) and (endq <= DatMat.nrows)) then begin
                tmpm := 0.0;
                for i := startq to endq do begin
                    jj := WrapTransect(i,DatMat.nrows);
                    if DatMat.IsNum[jj,1] then
                       case Wavelet of
                            1 : tmpm := tmpm + DatMat[jj,1] * HaarWavelet(SignedEucDist(i,x) / b);
                            2 : tmpm := tmpm + DatMat[jj,1] * FrenchTopHatWavelet(SignedEucDist(i,x) / b);
                            3 : tmpm := tmpm + DatMat[jj,1] * MexicanhatWavelet(SignedEucDist(i,x) / b);
                            4 : tmpm := tmpm + DatMat[jj,1] * MorletWavelet(SignedEucDist(i,x) / b);
                            5 : tmpm := tmpm + DatMat[jj,1] * SineWavelet(SignedEucDist(i,x) / b);
                       end
                    else good := false;
                end;
                if good then begin
                   WOutMat[b,x] := tmpm / b;
                   VOutMat[b,outcol] := VOutMat[b,outcol] + sqr(WOutMat[b,x]);
                   inc(cnt);
                end;
             end;
         end;
         if (cnt > 0) then VOutMat[b,outcol] := VOutMat[b,outcol] / cnt
         else VOutMat.IsEmpty[b,outcol] := true;
         ProgressIncrement;
     end;
     for x := 1 to DatMat.nrows do if ContinueProgress then begin
         POutMat[x,outcol] := 0.0;

         cnt := 0;
         for b := 1 to maxb do
             if WOutMat.IsNum[b,x] then begin
                POutMat[x,outcol] := POutMat[x,outcol] + sqr(WOutMat[b,x]);
                inc(cnt);
             end;
         if (cnt > 0) then POutMat[x,outcol] := POutMat[x,outcol] / cnt
         else POutMat.IsEmpty[x,outcol] := true;
     end;
     if ContinueProgress then
        for i := maxb+1 to maxxb do ProgressIncrement;
end;

procedure CalcWavelets(tmpDat : TpasBasicMatrix; VOutMat,POutMat,WOutMat : TpasMatrix;
          P3OutMat,W3OutMat : Tpas3DMatrix; W4OutMat : TpasDouble4Array;
          outcol,maxb : integer; Dim,Meth : byte; range : double;
          DoWrap : boolean; var W4Good : TpasBoolean4Array);
begin
     case Dim of
          1 : Calc_1DWavelet(Meth,TpasMatrix(tmpDat),VOutMat,POutMat,WOutMat,outcol,maxb,range,DoWrap);
          2 : Calc_2DWavelet(Meth,TpasMatrix(tmpDat),VOutMat,POutMat,W3OutMat,outcol,maxb,range);
          3 : Calc_3DWavelet(Meth,Tpas3DMatrix(tmpDat),VOutMat,P3OutMat,W4OutMat,outcol,maxb,range,W4Good);
     end;
end;

procedure WaveletAnalysis(DatMat : TpasMatrix; Dat3D : Tpas3DMatrix;
          dim, meth : integer; range, scale : double; DoRand : boolean;
          Niters : integer; ConfInt : double; RandScale, RandPos, RandScalePos,
          SaveOut : boolean; VName,PName,WName : string;
          IncX,IncY,IncZ : boolean; DoData : TpasBooleanArray;
          DoScalePlot,DoPosPlot,DoWPlot,DoAllPlot,DoWrap,DoBubble : boolean);
var
   ix,iy,iz,
   i,r,k,b,
   maxl,maxb,
   HiInd,LoInd,
   LoWInd,HiWInd,
   maxx,maxy,maxz : integer;
   DataSets : TObjectList;
   tmpDat : TpasBasicMatrix;
   IntOut,
   DoXY,DoXZ,DoYZ,
   DoX,DoY,DoZ,DoRow,DoCol : TpasBooleanArray;
   RandWH2dtmp,RandWL2dtmp,RandP3dtmp,
   W4OutMat,RandW4 : TpasDouble4Array;
   RWH2dInd,RWL2dInd,RP3dInd,
   RandWHtmp,RandWLtmp,RandP2Dtmp,
   RandP3High,RandW3Low,RandW3High,
   W3OutMat,P3OutMat,RandW3,RandP3 : Tpas3DMatrix;
   RandVtmp,RandPtmp,
   RWLind,RWHind,RPind,RVind,
   RandWHigh,RandWLow,
   RandPHigh,
   RandV,RandW,RandP,WOutMat,VOutMat,POutMat : TpasMatrix;
   VHeader : TpasTableHeader;
   p1,p2 : double;
   fstr,mname : string;
   SubText : TStringList;
   W4Good : TpasBoolean4Array;
   {$IFNDEF FPC}
   PlForm : TPlotForm;
   {$ENDIF}
begin
     if DoTimeStamp then StartTimeStamp('Wavelet Analysis');
     DoX := nil; DoY := nil; DoZ := nil; DoRow := nil; DoCol := nil;
     DoXY := nil; DoXZ := nil; DoYZ := nil;
     // figure out which subsets to analyze
     FindSubSets(DatMat,Dat3D,Dim,IncX,IncY,IncZ,DoData,DoRow,DoCol,DoX,DoY,DoZ,
        DoXY,DoXZ,DoYZ);
     // retrieve each individual data set to be analyzed
     DataSets := TObjectList.Create;
     RetrieveDataSets(DatMat,Dat3D,dim,DataSets,IncX,IncY,IncZ,DoCol,DoRow,
        DoXY,DoXZ,DoYZ,DoX,DoY,DoZ,maxl,maxx,maxy,maxz);

     // find maximum scale in any data set
     maxb := trunc(maxl * range / 100.0);
     if (maxb > maxl div 2) then maxb := maxl div 2;
     if (maxb < 2) then maxb := 2;


     if DoRand then k := niters + 1 else k := 1;
     ProgressRefresh(k*maxb,'Calculating Wavelets...');

     // W is two-tailed, V and P are one-tailed
     LoWInd := round((niters+1.0) * (1.0 - ConfInt) / 2.0);
     if (LoWInd < 1) then LoWInd := 1;
     HiWInd := round((niters+1.0) * (1.0 - (1.0 - ConfInt) / 2.0));
     if (HiWInd > niters+1) then HiWInd := niters + 1;
     HiInd := round((niters+1.0) * ConfInt);
     if (HiInd > niters+1) then HiInd := niters + 1;
     LoInd := round((niters+1.0) * (1.0 - ConfInt));

     // setup output matrix
     if DoRand and RandScale then VOutMat := TpasMatrix.Create(maxb,3)
     else VOutMat := TpasMatrix.Create(maxb,2);
     VOutMat.MatrixName := VName;
     VOutMat.ColLabel[1] := 'Scale';

     POutMat := nil; P3OutMat := nil;
     WOutMat := nil; W3OutMat := nil; W4OutMat := nil;
     RandP3High := nil; RandPHigh := nil;
     RandWLow := nil; RandWHigh := nil; RandW3High := nil; RandW3Low := nil;
     RandWLtmp := nil; RandWHtmp := nil; RWLind := nil; RWHind := nil;
     RWH2dInd := nil; RWL2dInd := nil;
     RandPtmp := nil; RandP2dtmp := nil; RPInd := nil;
     RandP3Dtmp := nil; RP3Dind := nil;

     case Dim of
          1 : begin
                   if DoRand and RandPos then POutMat := TpasMatrix.Create(maxl,3)
                   else POutMat := TpasMatrix.Create(maxl,2);
                   POutMat.ColLabel[1] := 'Position';
                   for ix := 1 to POutMat.nrows do
                       POutMat[ix,1] := ix * scale;
                   WOutMat := TpasMatrix.Create(maxb,maxl);
                   WOutMat.MatrixName := WName;
              end;
          2 : begin
                   POutMat := TpasMatrix.Create(maxy,maxx);
                   W3OutMat :=Tpas3DMatrix.Create(maxb,maxy,maxx);
                   W3OutMat.MatrixName := WName;
              end;
          3 : begin
                   P3OutMat := Tpas3DMatrix.Create(maxx,maxy,maxz);
                   SetLength(W4OutMat,maxb,maxx,maxy,maxz);
              end;
     end;
     if (POutMat <> nil) then POutMat.MatrixName := PName
     else P3OutMat.MatrixName := PName;


     VHeader := TpasTableHeader.Create;
     VHeader.AddBase('Scale');
     VHeader.AddBlank(1);
     // scale
     for b := 1 to maxb do begin
         VOutMat[b,1] := b * scale;
         VOutMat.RowLabel[b] := 'Scale ' + IntToStr(b);
     end;
       // method name
     case Dim of
          1 : mname := WaveletNames1d[meth] + ' Wavelet';
          2 : mname := WaveletNames2d[meth] + ' Wavelet';
          3 : mname := WaveletNames3d[meth] + ' Wavelet';
          else mname := 'Wavelet';
     end;

     p2 := 100.0 * (1.0 - ConfInt) / 2.0;
     p1 := 100.0 * (1.0 - ConfInt);
     i := 1;
     tmpDat := TpasBasicMatrix(DataSets[i-1]);
     ProgressCaption('Wavelet Analysis of ' + tmpDat.MatrixName);
     VOutMat.ColLabel[2] := tmpDat.MatrixName + ' ' + mname;
     if (Dim = 1) then POutMat.ColLabel[2] := tmpDat.MatrixName + ' ' + mname;
     if DoRand and RandScale then begin
        VHeader.AddBase('Variance');
        VHeader.AddBase(format('%g',[100.0-p1])+'%');
        VHeader.AddOther(2,3,3,'Null Distribution');
        VHeader.AddOther(3,2,3,tmpDat.MatrixName);
        VOutMat.ColLabel[3] := tmpDat.MatrixName + ' Rand '+format('%g',[100.0-p1])+'% Null';
     end else VHeader.AddBase(tmpDat.MatrixName);
     if DoRand and RandPos and (Dim = 1) then
        POutMat.ColLabel[3] := tmpDat.MatrixName + ' Rand '+format('%g',[100.0-p1])+'% Null';

     CalcWavelets(tmpDat,VOutMat,POutMat,WOutMat,P3OutMat,W3OutMat,W4OutMat,
            2,maxb,Dim,Meth,range,DoWrap,W4Good);

     // randomization test
     if ContinueProgress and DoRand then begin
        ProgressCaption('Randomization test...');
        RandV := TpasMatrix.Create(maxb,1);
        if RandScale then begin
           RandVtmp := TpasMatrix.Create(maxb,LoInd);
           RVInd := TpasMatrix.Create(maxb,1);
           for b := 1 to maxb do
               if VOutMat.IsNum[b,2] then begin
                  RandVtmp[b,1] := VoutMat[b,2];
                  RVind[b,1] := 1.0;
               end else RVind[b,1] := 0.0;
        end else begin
            RandVtmp := nil;
            RVInd := nil;
        end;

        RandP := nil; RandP3 := nil;
        RandW := nil; RandW3 := nil; RandW4 := nil;
        case Dim of
             1 : begin
                      RandP := TpasMatrix.Create(maxl,niters);
                      RandW := TpasMatrix.Create(maxb,maxl);
                      if RandPos then begin
                         RandPtmp := TpasMatrix.Create(maxl,LoInd);
                         RPInd := TpasMatrix.Create(maxl,1);
                         for ix := 1 to maxl do
                             if POutMat.IsNum[ix,2] then begin
                                RandPtmp[ix,1] := POutMat[ix,2];
                                RPind[ix,1] := 1.0;
                             end else RPind[ix,1] := 0.0;
                      end;
                      if RandScalePos then begin
                         RandWLtmp := Tpas3dMatrix.Create(maxb,maxl,LoWInd);
                         RandWHtmp := Tpas3dMatrix.Create(maxb,maxl,LoWInd);
                         RWLind := TpasMatrix.Create(maxb,maxl);
                         RWHind := TpasMatrix.Create(maxb,maxl);
                         for b := 1 to maxb do
                             for ix := 1 to maxl do
                                 if WOutMat.IsNum[b,ix] then begin
                                    RandWLtmp[b,ix,1] := WOutMat[b,ix];
                                    RandWHtmp[b,ix,1] := WOutMat[b,ix];
                                    RWLind[b,ix] := 1.0;
                                    RWHind[b,ix] := 1.0;
                                 end else begin
                                     RWLind[b,ix] := 0.0;
                                     RWHind[b,ix] := 0.0;
                                 end;
                         RandWHigh := TpasMatrix.Create(maxb,maxl);
                         RandWHigh.MatrixName := WName + ' Upper '+format('%g',[100.0-p2])+'% Null';
                         RandWLow := TpasMatrix.Create(maxb,maxl);
                         RandWLow.MatrixName := WName + ' Lower '+format('%g',[p2])+'% Null';
                      end;
                 end;
             2 : begin
                      RandP := TpasMatrix.Create(maxy,maxx);
                      RandW3 :=Tpas3DMatrix.Create(maxb,maxy,maxx);
                      if RandPos then begin
                         RandP2Dtmp := Tpas3dMatrix.Create(maxy,maxx,LoInd);
                         RPind := TpasMatrix.Create(maxy,maxx);
                         for ix := 1 to maxx do
                             for iy := 1 to maxy do
                                 if POutMat.IsNum[iy,ix] then begin
                                    RandP2Dtmp[iy,ix,1] := POutMat[iy,ix];
                                    RPind[iy,ix] := 1.0;
                                 end else RPind[iy,ix] := 0.0;
                         RandPHigh := TpasMatrix.Create(maxy,maxx);
                         RandPHigh.MatrixName := PName + ' Upper '+format('%g',[100.0-p1])+'% Null';
                      end;
                      if RandScalePos then begin
                         SetLength(RandWH2Dtmp,maxb+1,maxy+1,maxx+1,LoWInd+1);
                         SetLength(RandWL2Dtmp,maxb+1,maxy+1,maxx+1,LoWInd+1);
                         RWH2dInd := Tpas3dMatrix.Create(maxb,maxy,maxx);
                         RWL2dInd := Tpas3dMatrix.Create(maxb,maxy,maxx);
                         for b := 1 to maxb do
                             for ix := 1 to maxx do
                                 for iy := 1 to maxy do
                                     if W3OutMat.IsNum[b,iy,ix] then begin
                                        RandWL2Dtmp[b,iy,ix,1] := W3OutMat[b,iy,ix];
                                        RandWH2Dtmp[b,iy,ix,1] := W3OutMat[b,iy,ix];
                                        RWL2Dind[b,iy,ix] := 1.0;
                                        RWH2Dind[b,iy,ix] := 1.0;
                                     end else begin
                                         RWH2dInd[b,iy,ix] := 0.0;
                                         RWL2dInd[b,iy,ix] := 0.0;
                                     end;
                         RandW3High := Tpas3DMatrix.Create(maxb,maxy,maxx);
                         RandW3High.MatrixName := WName + ' Upper '+format('%g',[100.0-p2])+'% Null';
                         RandW3Low := Tpas3DMatrix.Create(maxb,maxy,maxx);
                         RandW3Low.MatrixName := WName + ' Lower '+format('%g',[p2])+'% Null';
                      end;
                 end;
             3 : begin
                      RandP3 := Tpas3DMatrix.Create(maxx,maxy,maxz);
                      SetLength(RandW4,maxb,maxx,maxy,maxz);
                      if RandPos then begin
                         SetLength(RandP3Dtmp,maxx+1,maxy+1,maxz+1,LoInd+1);
                         RP3dInd := Tpas3dMatrix.Create(maxx,maxy,maxx);
                         for ix := 1 to maxx do
                             for iy := 1 to maxy do
                                 for iz := 1 to maxz do
                                     if P3OutMat.IsNum[ix,iy,iz] then begin
                                        RandP3Dtmp[ix,iy,iz,1] := P3OutMat[ix,iy,iz];
                                        RP3Dind[ix,iy,iz] := 1.0;
                                     end else RP3dInd[ix,iy,iz] := 0.0;
                         RandP3High := Tpas3DMatrix.Create(maxx,maxy,maxz);
                         RandP3High.MatrixName := PName + ' Upper '+format('%g',[100.0-p1])+'% Null';
                      end;
                 end;
        end;

        for r := 1 to niters do if ContinueProgress then begin
            case Dim of
                 1 : RandomizeTransects(TpasMatrix(tmpDat),nil,1,seed);
                 2 : RandomizeSurfaces(TpasMatrix(tmpDat),nil,seed);
                 3 : RandomizeVolumes(Tpas3DMatrix(tmpDat),nil,seed);
            end;
            CalcWavelets(tmpDat,RandV,RandP,RandW,RandP3,RandW3,RandW4,1,
                    maxb,Dim,Meth,range,DoWrap,W4Good);
            {CalcWavelets(tmpDat,RandV,RandP,RandW,RandP3,RandW3,RandW4,r,
                    maxb,Dim,Meth,range,seed,DoWrap,W4Good);}

            // sort values
            if RandScale then
               for b := 1 to maxb do
                   if RandV.IsNum[b,1] then
                      RVInd[b,1] := AdjustPermValues(RandV[b,1],RandVtmp,
                                       b,trunc(RVInd[b,1]),true);
            case Dim of
                 1 : begin
                          if RandPos then
                             for ix := 1 to maxl do
                                 if RandP.IsNum[ix,1] then
                                    RPInd[ix,1] := AdjustPermValues(RandP[ix,1],RandPtmp,
                                       ix,trunc(RPInd[ix,1]),true);
                          if RandScalePos then
                             for b := 1 to maxb do
                                 for ix := 1 to maxl do
                                     if RandW.IsNum[b,ix] then begin
                                        RWLInd[b,ix] := AdjustPermValues(RandW[b,ix],RandWLtmp,
                                           b,ix,trunc(RWLInd[b,ix]),false);
                                        RWHInd[b,ix] := AdjustPermValues(RandW[b,ix],RandWHtmp,
                                           b,ix,trunc(RWHInd[b,ix]),true);
                                     end;
                     end;
                 2 : begin
                          if RandPos then
                             for ix := 1 to maxx do
                                 for iy := 1 to maxy do
                                     if RandP.IsNum[iy,ix] then
                                        RPInd[iy,ix] := AdjustPermValues(RandP[iy,ix],RandP2Dtmp,
                                            iy,ix,trunc(RPInd[iy,ix]),true);
                          if RandScalePos then
                             for b := 1 to maxb do
                                 for ix := 1 to maxx do
                                     for iy := 1 to maxy do
                                         if RandW3.IsNum[b,iy,ix] then begin
                                            RWL2dInd[b,iy,ix] := AdjustPermValues(RandW3[b,iy,ix],RandWL2Dtmp,
                                                b,iy,ix,trunc(RWL2dInd[b,iy,ix]),LoWInd,false);
                                            RWH2dInd[b,iy,ix] := AdjustPermValues(RandW3[b,iy,ix],RandWH2Dtmp,
                                                b,iy,ix,trunc(RWH2dInd[b,iy,ix]),LoWInd,true);
                                         end;
                     end;
                 3 : if RandPos then
                        for ix := 1 to maxx do
                            for iy := 1 to maxy do
                                for iz := 1 to maxz do
                                    if RandP3.IsNum[ix,iy,iz] then
                                       RP3dInd[ix,iy,iz] := AdjustPermValues(RandP3[ix,iy,iz],RandP3Dtmp,
                                          ix,iy,iz,trunc(RP3dInd[ix,iy,iz]),LoInd,true);
            end;
        end;
        RandV.Free;
        if (RandP <> nil) then RandP.Free;
        if (RandP3 <> nil) then RandP3.Free;
        if (RandW <> nil) then RandW.Free;
        if (RandW3 <> nil) then RandW3.Free;
        RandW4 := nil;
        
        if RandScale and ContinueProgress then begin
           for b := 1 to maxb do
               if RandVtmp.IsNum[b,trunc(RVInd[b,1])] then
                  VOutMat[b,3] := RandVtmp[b,trunc(RVInd[b,1])];
           RVInd.Free;
           RandVtmp.Free;
        end;

        if RandPos and ContinueProgress then begin
           case Dim of
                1 : for ix := 1 to maxl do
                        if RandPtmp.IsNum[ix,trunc(RPInd[ix,1])] then
                           POutMat[ix,3] := RandPtmp[ix,trunc(RPInd[ix,1])];
                2 : for ix := 1 to maxx do
                        for iy := 1 to maxy do
                            if RandP2Dtmp.IsNum[iy,ix,trunc(RPInd[iy,ix])] then
                               RandPHigh[iy,ix] := RandP2Dtmp[iy,ix,trunc(RPInd[iy,ix])];
                3 : for ix := 1 to maxx do
                        for iy := 1 to maxy do
                            for iz := 1 to maxz do
                                if (RP3dInd[ix,iy,iz] > 0) then
                                    RandP3High[ix,iy,iz] := RandP3Dtmp[ix,iy,iz,trunc(RP3dInd[ix,iy,iz])];
           end;

           if (RandPtmp <> nil) then RandPtmp.Free;
           if (RandP2Dtmp <> nil) then RandP2Dtmp.Free;
           if (RPInd <> nil) then RPind.Free;
           if (RP3DInd <> nil) then RP3Dind.Free;
           RandP3Dtmp := nil;
        end;


        if RandScalePos then if ContinueProgress then begin
           case Dim of
                1 : for ix := 1 to maxl do
                        for iy := 1 to maxb do begin
                            if RandWLtmp.IsNum[iy,ix,trunc(RWLInd[iy,ix])]then
                                    RandWLow[iy,ix] := RandWLtmp[iy,ix,trunc(RWLInd[iy,ix])];
                            if RandWHtmp.IsNum[iy,ix,trunc(RWHInd[iy,ix])]then
                                    RandWHigh[iy,ix] := RandWHtmp[iy,ix,trunc(RWHInd[iy,ix])];
                        end;
                2 : for b := 1 to maxb do
                        for ix := 1 to maxx do
                            for iy := 1 to maxy do begin
                                if (RWL2dInd[b,iy,ix] > 0) then
                                    RandW3Low[b,iy,ix] := RandWL2Dtmp[b,iy,ix,trunc(RWL2dInd[b,iy,ix])];
                                if (RWH2dInd[b,iy,ix] > 0) then
                                    RandW3High[b,iy,ix] := RandWH2Dtmp[b,iy,ix,trunc(RWH2dInd[b,iy,ix])];
                            end;
           end;
           if (RandWLtmp <> nil) then RandWLtmp.Free;
           if (RandWHtmp <> nil) then RandWHtmp.Free;
           if (RWLind <> nil) then RWLind.Free;
           if (RWHind <> nil) then RWHind.Free;
           if (RWL2Dind <> nil) then RWL2Dind.Free;
           if (RWH2Dind <> nil) then RWH2Dind.Free;
           RandWL2Dtmp := nil;
           RandWH2Dtmp := nil;
        end;
        
     end;
     
     // Output
     if ContinueProgress then begin
        SetLength(IntOut,VOutMat.ncols);
        for i := 1 to VOutMat.ncols do IntOut[i-1] := false;
        SubText := TStringList.Create;
        fstr := FormatFloatStr(OutputDecs);
        if (DatMat <> nil) then
           SubText.Add('Data matrix: ' + DatMat.MatrixName)
        else SubText.Add('Data matrix: ' + Dat3D.MatrixName);
        SubText.Add(IntToStr(Dim) + 'D analysis');
        if (Dim = 1) and DoWrap then SubText.Add('  Wrapping transects');
        SubText.Add('Wavelet: '+mname);
        SubText.Add('Maximum scale = ' + format(fstr,[range]) + '%');
        SubText.Add('Scale multiplier = ' + format(fstr,[scale]));
        if DoRand then begin
           SubText.Add('Permutation tests based on ' + IntToStr(niters) + ' permutations');
           if RandScale then SubText.Add('  Permutation test for scalar variance');
           if RandPos then SubText.Add('  Permutation test for positional variance');
           if RandScalePos then SubText.Add('  Permutation test for positional x scale variance');
        end;
        WriteOutputTable(VOutMat,IntOut,VHeader,'Wavelet Analysis',SubText);
        SubText.Free;
        IntOut := nil;
     end;
     
     {$IFNDEF FPC}
     // Plot V
     if ContinueProgress and DoScalePlot then begin
        MainForm.CreatePlotForm(PlForm);
        with PLForm do begin
             Caption := mname + ': Scale Variance';
             if DoRand and RandScale then
                DrawProfileVarLimits(VOutMat,1,2,0,0,0,0,3,
                   'Scale','Variance',true,VOutMat.ColLabel[2],false)
             else DrawProfileVarLimits(VOutMat,1,2,0,0,0,0,0,'Scale',
                  'Variance',true,VOutMat.ColLabel[2],false);
             Show;
        end;
     end;

     // Plot P
     if ContinueProgress and DoPosPlot then begin
        MainForm.CreatePlotForm(PlForm);
        case Dim of
             1 : with PLForm do begin
                      Caption := mname + ': Positional Variance';
                      if DoRand and RandPos then
                         DrawProfileVarLimits(POutMat,1,2,0,0,0,0,3,'Position',
                            'Variance',true,POutMat.ColLabel[2],false)
                      else DrawProfileVarLimits(POutMat,1,2,0,0,0,0,0,'Position',
                            'Variance',true,POutMat.ColLabel[2],false);
                      Show;
                 end;
             2 : with PLForm do begin
                      Caption := mname + ': Positional Variance';
                      if DoRand and RandPos then
                         DrawConfidenceSurface(POutMat,RandPHigh,nil,'X','Y',true,0,0,scale,scale,true)
                      else DrawSurfacePlot(POutMat,'X','Y',true,0,0,scale,scale,0,true);
                      Show;
                 end;
             3 : with PLForm do begin
                      Caption := mname + ': Positional Variance';
                      if DoRand and RandPos then begin
                         if DoBubble then
                            Draw3DCubeBubbleConfidencePlot(P3OutMat,RandP3High,nil,0.0,0.0,0.0,scale,scale,scale)
                         else Draw3DCubeSurfaceConfidencePlot(P3OutMat,RandP3High,nil,1,0.0,0.0,0.0,scale,scale,scale);
                      end else if DoBubble then Draw3DCubeBubblePlot(P3OutMat,0.0,0.0,0.0,scale,scale,scale)
                      else Draw3DCubeSurfacePlot(P3OutMat,0.0,0.0,0.0,scale,scale,scale);
                      Show;
                 end;
        end;
     end;

     // Plot W
     if ContinueProgress and DoWPLot and (Dim < 3) then begin
        MainForm.CreatePlotForm(PlForm);
        case Dim of
             1 : with PLForm do begin
                      Caption := mname + ': Position x Scale Variance';
                      if DoRand and RandScalePos then
                         DrawConfidenceSurface(WOutMat,RandWHigh,RandWLow,'Position','Scale',false,0,0,scale,scale,true)
                      else DrawSurfacePlot(WOutMat,'Position','Scale',false,0,0,scale,scale,0,true);
                      Show;
                 end;
             2 : with PLForm do begin
                      Caption := mname + ': Position x Scale Variance';
                      if DoRand and RandScalePos then begin
                         if DoBubble then
                            Draw3DCubeBubbleConfidencePlot(W3OutMat,RandW3High,RandW3Low,0.0,0.0,0.0,1.0,scale,scale)
                         else Draw3DCubeSurfaceConfidencePlot(W3OutMat,RandW3High,RandW3Low,1,0.0,0.0,0.0,1.0,scale,scale);
                      end else if DoBubble then Draw3DCubeBubblePlot(W3OutMat,0.0,0.0,0.0,1.0,scale,scale)
                      else Draw3DCubeSurfacePlot(W3OutMat,0.0,0.0,0.0,1.0,scale,scale);
                      Show;
                 end;
        end;

     end;

     if ContinueProgress and DoAllPlot and (Dim = 1) then begin
        MainForm.CreatePlotForm(PlForm);
        with PLForm do begin
             Caption := mname;
             Draw1DWaveletCombo(VOutMat,POutMat,WOutMat,RandWHigh,RandWLow,scale,
                RandScale,RandPos,RandScalePos);
             Show;
        end;

     end;
     {$ENDIF}

     // Save matrix
     if ContinueProgress and SaveOut then begin
        Data_AddData(VOutMat);
        OutputAddLine('Scalar variance output saved to "'+VOutMat.MatrixName+'".');
        if (POutMat <> nil) then begin
           Data_AddData(POutMat);
           OutputAddLine('Positional variance output saved to "'+POutMat.MatrixName+'".');
        end;
        if (RandPHigh <> nil) then begin
           Data_AddData(RandPHigh);
           OutputAddLine('Positional variance upper null distribution saved to "'+RandPHigh.MatrixName+'".');
        end;
        if (P3OutMat <> nil) then begin
           Data_AddData(P3OutMat);
           OutputAddLine('Positional variance output saved to "'+P3OutMat.MatrixName+'".');
        end;
        if (RandP3High <> nil) then begin
           Data_AddData(RandP3High);
           OutputAddLine('Positional variance upper null distribution saved to "'+RandP3High.MatrixName+'".');
        end;
        if (WOutMat <> nil) then begin
           Data_AddData(WOutMat);
           OutputAddLine('Scale x Positional variance output saved to "'+WOutMat.MatrixName+'".');
        end;
        if (W3OutMat <> nil) then begin
           Data_AddData(W3OutMat);
           OutputAddLine('Scale x Positional variance output saved to "'+W3OutMat.MatrixName+'".');
        end;
        if (RandWLow <> nil) then begin
           Data_AddData(RandWLow);
           OutputAddLine('Scale x Positional variance lower null distribution saved to "'+RandWLow.MatrixName+'".');
        end;
        if (RandWHigh <> nil) then begin
           Data_AddData(RandWHigh);
           OutputAddLine('Scale x Positional variance upper null distribution saved to "'+RandWHigh.MatrixName+'".');
        end;
        if (RandW3Low <> nil) then begin
           Data_AddData(RandW3Low);
           OutputAddLine('Scale x Positional variance lower null distribution saved to "'+RandW3Low.MatrixName+'".');
        end;
        if (RandW3High <> nil) then begin
           Data_AddData(RandW3High);
           OutputAddLine('Scale x Positional variance upper null distribution saved to "'+RandW3High.MatrixName+'".');
        end;
        OutputAddBlankLine;
     end else begin
         VOutMat.Free;
         if (POutMat <> nil) then POutMat.Free;
         if (P3OutMat <> nil) then P3OutMat.Free;
         if (WOutMat <> nil) then WOutMat.Free;
         if (W3OutMat <> nil) then W3OutMat.Free;
         //if (RandPLow <> nil) then RandPLow.Free;
         if (RandPHigh <> nil) then RandPHigh.Free;
         if (RandP3High <> nil) then RandP3High.Free;
         if (RandWLow <> nil) then RandWLow.Free;
         if (RandWHigh <> nil) then RandWHigh.Free;
         if (RandW3Low <> nil) then RandW3Low.Free;
         if (RandW3High <> nil) then RandW3High.Free;
     end;

     // Free Memory
     W4OutMat := nil; W4Good := nil;
     VHeader.Free;
     DoX := nil; DoY := nil; DoZ := nil; DoRow := nil; DoCol := nil;
     DoXY := nil; DoXZ := nil; DoYZ := nil;
     DataSets.Free;
     ProgressClose;
     if DoTimeStamp then EndTimeStamp;
end;

"""
