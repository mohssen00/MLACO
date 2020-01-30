function [cosine] = Cosine(data)
cosine = squareform(1-pdist(data,'cosine')) + eye(size(data,1));
cosine = abs(cosine);
cosine(cosine == 0) = 0.001;
end