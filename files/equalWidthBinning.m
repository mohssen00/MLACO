function [result] = equalWidthBinning(dataset, targets)

[nRow, nCol] = size(dataset);
[~, nLabel] = size(targets);
nCol = nCol - nLabel;

result = zeros(nRow, nCol);

    for i=1:nCol
        width=(max(dataset(i, :)) - min (dataset(i, :))) / 2; 
        result(dataset(i, :) > width, i) = 1; 
    end
result = [result targets];
end
