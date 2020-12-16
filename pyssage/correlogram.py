from math import sqrt, pi, degrees
from typing import Optional, Tuple
import numpy
import scipy.stats
from pyssage.connections import Connections
from pyssage.utils import create_output_table, check_for_square_matrix
import pyssage.mantel


def check_variance_assumption(x: Optional[str]) -> None:
    valid = ("random", "normal", None)
    if x not in valid:
        raise ValueError(x + " is not a valid variance assumption. Valid values are: " +
                         ", ".join((str(i) for i in valid)))


def morans_i(y: numpy.ndarray, weights: Connections, alt_weights: Optional[numpy.ndarray] = None,
             variance: Optional[str] = "random"):
    check_variance_assumption(variance)
    n = len(y)
    mean_y = numpy.average(y)
    dev_y = y - mean_y  # deviations from mean
    w = weights.as_binary()
    if alt_weights is not None:  # multiply to create non-binary weights, if necessary
        w = w * alt_weights
    sumyij = numpy.sum(numpy.outer(dev_y, dev_y) * w, dtype=numpy.float64)
    sumy2 = numpy.sum(numpy.square(dev_y), dtype=numpy.float64)  # sum of squared deviations from mean
    sumw = numpy.sum(w, dtype=numpy.float64)  # sum of weight matrix
    sumw2 = sumw**2
    moran = n * sumyij / (sumw * sumy2)
    expected = -1 / (n - 1)
    if variance is None:
        sd, z, p = None, None, None
    else:
        s1 = numpy.sum(numpy.square(w + numpy.transpose(w)), dtype=numpy.float64) / 2
        s2 = numpy.sum(numpy.square(numpy.sum(w, axis=0) + numpy.sum(w, axis=1)), dtype=numpy.float64)
        if variance == "normal":
            v = ((n**2 * s1) - n*s2 + 3*sumw2) / ((n**2 - 1) * sumw2)
        else:  # random
            b2 = n * numpy.sum(numpy.power(dev_y, 4), dtype=numpy.float64) / (sumy2**2)
            v = ((n*((n**2 - 3*n + 3)*s1 - n*s2 + 3*sumw2) - b2*((n**2 - n)*s1 - 2*n*s2 + 6*sumw2)) /
                 ((n - 1)*(n - 2)*(n - 3)*sumw2)) - 1/(n - 1)**2
        sd = sqrt(v)  # convert to standard dev
        z = abs(moran - expected) / sd
        p = scipy.stats.norm.sf(z)*2  # two-tailed test

    return weights.min_scale, weights.max_scale, weights.n_pairs(), expected, moran, sd, z, p


def gearys_c(y: numpy.ndarray, weights: Connections, alt_weights: Optional[numpy.ndarray] = None,
             variance: Optional[str] = "random"):
    check_variance_assumption(variance)
    n = len(y)
    mean_y = numpy.average(y)
    dev_y = y - mean_y  # deviations from mean
    w = weights.as_binary()
    if alt_weights is not None:  # multiply to create non-binary weights, if necessary
        w *= alt_weights
    sumdif2 = numpy.sum(numpy.square(w * (dev_y[:, numpy.newaxis] - dev_y)), dtype=numpy.float64)
    sumy2 = numpy.sum(numpy.square(dev_y), dtype=numpy.float64)  # sum of squared deviations from mean
    sumw = numpy.sum(w, dtype=numpy.float64)  # sum of weight matrix
    sumw2 = sumw**2
    geary = (n - 1) * sumdif2 / (2 * sumw * sumy2)
    if variance is None:
        sd, z, p = None, None, None
    else:
        s1 = numpy.sum(numpy.square(w + numpy.transpose(w)), dtype=numpy.float64) / 2
        s2 = numpy.sum(numpy.square(numpy.sum(w, axis=0) + numpy.sum(w, axis=1)), dtype=numpy.float64)
        if variance == "normal":
            v = ((2*s1 + s2)*(n - 1) - 4*sumw2) / (2*(n + 1)*sumw2)
        else:  # random
            b2 = n * numpy.sum(numpy.power(dev_y, 4), dtype=numpy.float64) / (sumy2 ** 2)
            nn2n3 = n * (n - 2) * (n - 3)
            v = ((n - 1)*s1*(n**2 - 3*n + 3 - (n - 1)*b2) / (nn2n3*sumw2) -
                 (n - 1)*s2*(n**2 + 3*n - 6 - (n**2 - n + 2)*b2) / (4*nn2n3*sumw2) +
                 (n**2 - 3 - b2*(n - 1)**2) / nn2n3)
        sd = sqrt(v)  # convert to standard dev
        z = abs(geary - 1) / sd
        p = scipy.stats.norm.sf(z)*2  # two-tailed test

    return weights.min_scale, weights.max_scale, weights.n_pairs(), 1, geary, sd, z, p


def mantel_correl(y: numpy.ndarray, weights: Connections, alt_weights: Optional[numpy.ndarray] = None,
                  variance: Optional[str] = "random"):
    r, p_value, tmp_text, _, _, _, z, sd = pyssage.mantel.mantel(y, weights.as_reverse_binary(), [])

    # return r, p_value, output_text, permuted_left_p, permuted_right_p, permuted_two_p, z_score, observed_std

    # # check_variance_assumption(variance)
    # n = len(y)
    #
    # # def mantel(input_matrix1, input_matrix2, partial, permutations: int = 0,
    # #            tail: str = "both") -> Tuple[float, float, list, float, float, float]:
    #
    # mean_y = numpy.average(y)
    # dev_y = y - mean_y  # deviations from mean
    # w = weights.as_binary()
    # if alt_weights is not None:  # multiply to create non-binary weights, if necessary
    #     w = w * alt_weights
    # sumyij = numpy.sum(numpy.outer(dev_y, dev_y) * w, dtype=numpy.float64)
    # sumy2 = numpy.sum(numpy.square(dev_y), dtype=numpy.float64)  # sum of squared deviations from mean
    # sumw = numpy.sum(w, dtype=numpy.float64)  # sum of weight matrix
    # sumw2 = sumw**2
    # moran = n * sumyij / (sumw * sumy2)
    # expected = -1 / (n - 1)
    # if variance is None:
    #     sd, z, p = None, None, None
    # else:
    #     s1 = numpy.sum(numpy.square(w + numpy.transpose(w)), dtype=numpy.float64) / 2
    #     s2 = numpy.sum(numpy.square(numpy.sum(w, axis=0) + numpy.sum(w, axis=1)), dtype=numpy.float64)
    #     if variance == "normal":
    #         v = ((n**2 * s1) - n*s2 + 3*sumw2) / ((n**2 - 1) * sumw2)
    #     else:  # random
    #         b2 = n * numpy.sum(numpy.power(dev_y, 4), dtype=numpy.float64) / (sumy2**2)
    #         v = ((n*((n**2 - 3*n + 3)*s1 - n*s2 + 3*sumw2) - b2*((n**2 - n)*s1 - 2*n*s2 + 6*sumw2)) /
    #              ((n - 1)*(n - 2)*(n - 3)*sumw2)) - 1/(n - 1)**2
    #     sd = sqrt(v)  # convert to standard dev
    #     z = abs(moran - expected) / sd
    #     p = scipy.stats.norm.sf(z)*2  # two-tailed test

    return weights.min_scale, weights.max_scale, weights.n_pairs(), 0, r, sd, z, p_value


def correlogram(data: numpy.ndarray, dist_class_connections: list, metric: morans_i,
                variance: Optional[str] = "random"):
    if metric == morans_i:
        metric_title = "Moran's I"
        exp_format = "f"
    elif metric == gearys_c:
        metric_title = "Geary's c"
        exp_format = "d"
    elif metric == mantel_correl:
        metric_title = "Mantel"
        exp_format = "d"
    else:
        metric_title = ""
        exp_format = ""
    output = []
    for dc in dist_class_connections:
        output.append(metric(data, dc, variance=variance))

    # create basic output text
    output_text = list()
    output_text.append(metric_title + " Correlogram")
    output_text.append("")
    output_text.append("# of data points = {}".format(len(data)))
    if variance is not None:
        output_text.append("Distribution assumption = {}".format(variance))
    output_text.append("")
    col_headers = ("Min dist", "Max dist", "# pairs", "Expected", metric_title, "SD", "Z", "Prob")
    col_formats = ("f", "f", "d", exp_format, "f", "f", "f", "f")
    create_output_table(output_text, output, col_headers, col_formats)

    return output, output_text


def bearing_correlogram(data: numpy.ndarray, dist_class_connections: list, angles: numpy.ndarray, n_bearings: int = 18,
                        metric=morans_i, variance: Optional[str] = "random"):
    if metric == morans_i:
        metric_title = "Moran's I"
        exp_format = "f"
    elif metric == gearys_c:
        metric_title = "Geary's c"
        exp_format = "d"
    else:
        metric_title = ""
        exp_format = ""

    # calculate bearings and bearing weight matrices
    bearings = []
    bearing_weights = []
    for b in range(n_bearings):
        a = b * pi / n_bearings
        bearings.append(a)
        bearing_weights.append(numpy.square(numpy.cos(angles - a)))

    output = []
    for i, b in enumerate(bearing_weights):
        for dc in dist_class_connections:
            tmp_out = list(metric(data, dc, alt_weights=b, variance=variance))
            tmp_out.insert(2, degrees(bearings[i]))
            output.append(tmp_out)

    # create basic output text
    output_text = list()
    output_text.append(metric_title + " Bearing Correlogram")
    output_text.append("")
    output_text.append("# of data points = {}".format(len(data)))
    if variance is not None:
        output_text.append("Distribution assumption = {}".format(variance))
    output_text.append("")
    col_headers = ("Min dist", "Max dist", "Bearing", "# pairs", "Expected", metric_title, "SD", "Z", "Prob")
    col_formats = ("f", "f", "f", "d", exp_format, "f", "f", "f", "f")
    create_output_table(output_text, output, col_headers, col_formats)

    return output, output_text


def windrose_sectors_per_annulus(segment_param: int, annulus: int) -> int:
    return (segment_param*(annulus+1) - 2) // 2


def create_windrose_connections(distances: numpy.ndarray, angles: numpy.ndarray, annulus: int, sector: int,
                                a: int, c: float, d: float, e: float) -> Tuple[Connections, float, float]:
    n = check_for_square_matrix(distances)
    output = Connections(n)
    output.min_scale = c*annulus**2 + d*annulus + e
    output.max_scale = c*(annulus+1)**2 + d*(annulus+1) + e
    sector_breadth = pi / windrose_sectors_per_annulus(a, annulus)
    sector_min = sector*sector_breadth
    sector_max = (sector + 1)*sector_breadth
    for i in range(n):
        for j in range(i):
            if (output.min_scale <= distances[i, j] < output.max_scale) and (sector_min <= angles[i, j] < sector_max):
                output.store(i, j)
    return output, sector_min, sector_max


def windrose_correlogram(data: numpy.ndarray, distances: numpy.ndarray, angles: numpy.ndarray,
                         radius_c: float, radius_d: float, radius_e: float, segment_param: int = 4,
                         min_pairs: int = 21, metric=morans_i, variance: Optional[str] = "random"):
    if metric == morans_i:
        metric_title = "Moran's I"
        exp_format = "f"
    elif metric == gearys_c:
        metric_title = "Geary's c"
        exp_format = "d"
    else:
        metric_title = ""
        exp_format = ""

    # number of annuli and sectors
    maxdist = numpy.max(distances)
    n_annuli = 0
    while (radius_c*n_annuli**2 + radius_d*n_annuli + radius_e) <= maxdist:
        n_annuli += 1
    if n_annuli > 7:
        n_annuli = 7  # maximum annuli is 7

    output = []
    all_output = []
    # all_output is needed for graphing the output *if* we want to include those sectors with too few pairs, but
    # still more than zero
    for annulus in range(n_annuli):
        for sector in range(windrose_sectors_per_annulus(segment_param, annulus)):
            connection, min_ang, max_ang = create_windrose_connections(distances, angles, annulus, sector,
                                                                       segment_param, radius_c, radius_d, radius_e)
            np = connection.n_pairs()
            if np >= min_pairs:
                tmp_out = list(metric(data, connection, variance=variance))
                # add sector angles to output
                tmp_out.insert(2, degrees(min_ang))
                tmp_out.insert(3, degrees(max_ang))
                output.append(tmp_out)
                all_output.append(tmp_out)
            else:
                # using -1 for the probability as an indicator that nothing was calculated
                tmp_out = [connection.min_scale, connection.max_scale, degrees(min_ang), degrees(max_ang),
                           np, 0, 0, 0, 0, -1]
                all_output.append(tmp_out)

    # create basic output text
    output_text = list()
    output_text.append(metric_title + " Windrose Correlogram")
    output_text.append("")
    output_text.append("# of data points = {}".format(len(data)))
    output_text.append("")
    output_text.append("Sector parameter A = {}".format(segment_param))
    output_text.append("Distance parameters C = {:0.5f}, D = {:0.5f}, E = {:0.5f}".format(radius_c, radius_d, radius_e))
    output_text.append("")
    if variance is not None:
        output_text.append("Distribution assumption = {}".format(variance))

    output_text.append("")
    col_headers = ("Min dist", "Max dist", "Min angle", "Max angle", "# pairs", "Expected", metric_title, "SD",
                   "Z", "Prob")
    col_formats = ("f", "f", "f", "f", "d", exp_format, "f", "f", "f", "f")
    create_output_table(output_text, output, col_headers, col_formats)

    return output, output_text, all_output


def bearing(data: numpy.ndarray, distances: numpy.ndarray, angles: numpy.ndarray, nbearings: int):
    angle_width = pi / nbearings
    output = []
    for a in range(nbearings):
        test_angle = a * angle_width
        b_matrix = distances * numpy.square(numpy.cos(angles - test_angle))
        r, p_value, _, _, _, _, _, _ = pyssage.mantel.mantel(data, b_matrix, [])
        output.append([a*180/nbearings, r, p_value])

    # create basic output text
    output_text = list()
    output_text.append("Bearing Analysis")
    output_text.append("")
    output_text.append("Tested {} vectors".format(nbearings))
    output_text.append("")
    col_headers = ("Bearing", "Correlation", "Prob")
    col_formats = ("f", "f", "f")
    create_output_table(output_text, output, col_headers, col_formats)
    return output, output_text


"""
procedure Bearing(DatMat,DistMat : TpasSymmetricMatrix; AngMat : TpasAngleMatrix;
          n : integer; DoPlot, DoSave : boolean; SaveName : string;
          DoRand : boolean; niter : integer);
var
   OutMat : TpasMatrix;
   Header : TpasTableHeader;
   tmpD : TpasSymmetricMatrix;
   i,j,a : integer;
   testv,ang,r,fang : double;
   p,rp,lp,tp : extended;
   tmpList : TList;
   IntOut : TpasBooleanArray;
   SubText : TStringList;
   {$IFNDEF FPC}PlForm : TPlotForm;{$ENDIF}
begin
     if DoTimeStamp then StartTimeStamp('Bearing');
     if DoRand then OutMat := TpasMatrix.Create(n,4)
     else OutMat := TpasMatrix.Create(n,3);
     OutMat.MatrixName := SaveName;
     OutMat.ColLabel[1] := 'Bearing';
     OutMat.ColLabel[2] := 'Correlation';
     Outmat.ColLabel[3] := 'Asymp Prob';
     if DoRand then OutMat.ColLabel[4] := 'Perm Prob';
     for a := 1 to n do OutMat.RowLabel[a] := 'Bearing ' + IntToStr(a);

     tmpD := TpasSymmetricMatrix.Create(DatMat.N);

     ProgressRefresh(n,'Bearing Analysis...');
     ProgressShow;
     fang := Pi / n;
     tmpList := TList.Create;

     for a := 1 to n do
         if ContinueProgress then begin
            testv := (a - 1.0) * fang;
            for i := 1 to DistMat.N do
                for j := 1 to DistMat.N do
                    if (i <> j) then
                       if not AngMat.IsEmpty[i,j] and not DistMat.IsEmpty[i,j] then begin
                          // Calculate iteration angle
                          ang := AngMat[i,j];
                          while (ang > pi) do ang := ang - pi;
                          tmpD[i,j] := DistMat[i,j] * sqr(cos(ang - testv));
                       end else tmpD.IsEmpty[i,j] := true;
            Mantel(DatMat,tmpD,tmpList,DoRand,niter,false,true,r,p,rp,lp,tp);
            OutMat[a,1] := (a - 1.0) * 180.0 / n;
            OutMat[a,2] := r;
            OutMat[a,3] := p;
            if DoRand then OutMat[a,4] := tp;
            ProgressIncrement;
         end;
     tmpList.Free;
     tmpD.Free;

     // Output
     if ContinueProgress then begin
        Header := TpasTableHeader.Create;
        SetLength(IntOut,OutMat.ncols);
        for i := 1 to OutMat.ncols do IntOut[i-1] := false;
        Header.AddBase('Bearing');
        Header.AddBase('Correlation');
        if DoRand then begin
           Header.AddBase('Asymp');
           Header.AddBase('Perm');
           Header.AddOther(2,3,4,'Probability');
        end else Header.AddBase('Prob');
        SubText := TStringList.Create;
        SubText.Add('Data Distance Matrix: ' + DatMat.MatrixName);
        SubText.Add('Geographic Distance Matrix: ' + DistMat.MatrixName);
        SubText.Add('Angle Matrix : ' + AngMat.MatrixName);
        SubText.Add('Tested ' + IntToStr(n) + ' vectors');
        if DoRand then
           SubText.Add('Permutation test based on ' + IntToStr(niter) + ' permutations.');
        WriteOutputTable(OutMat,IntOut,Header,'Bearing Analysis',SubText);
        SubText.Free;
        Header.Free;
        IntOut := nil;
     end;

     {$IFNDEF FPC}
     if ContinueProgress and DoPlot then begin
        MainForm.CreatePlotForm(PlForm);
        with PLForm do begin
             Caption := 'Bearing Analysis Plot';
             DrawFixedNull(OutMat,1,0.0,'Null');
             if DoRand then
                DrawSigPointProfile(OutMat,1,1,2,4,-1,'Bearing','Correlation',true,false,0.05)
             else DrawSigPointProfile(OutMat,1,1,2,3,-1,'Bearing','Correlation',true,false,0.05);
             Show;
        end;
     end;
     {$ENDIF}

     if ContinueProgress and DoSave then begin
        Data_AddData(OutMat);
        OutputAddLine('Output saved to "' + OutMat.MatrixName + '".');
        OutputAddBlankLine;
     end else OutMat.Free;

     ProgressClose;
     if DoTimeStamp then EndTimeStamp;
end;
"""
