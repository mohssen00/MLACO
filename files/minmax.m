function [ data ] = minmax( data )
%MINMAX Summary of this function goes here
%   Detailed explanation goes here

data = (data - min(data)) ./ ( max(data) - min(data) );
end

