function [pheromones] = UFSACO2(X, initPheromone, nCycle, decayRate, featureCorrelation, featureLabelCorrelation, heuristicMethod)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%       X                           - n × d matrix, d dimensional training set with n patterns
%       initPheromone               - featrues initial pheromone matrix.
%       nCycle                      - maximum number of cycles that algorithm repeated.
%       decayRate                   - define the decay rate of the pheromone on each feature.
%       featureCorrelation          - correlation matrix of pair features.
%       featureLabelCorrelation     - correlation of features and class labels
%       heuristicMethod             - the method for calculation heuristic
% Output:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[~, nFeautures] = size(X);
pheromones = initPheromone;

% nAnt - define the number of agents (number of ants). 
% if nFeautures > 50
%     nAnt = 50;
% else
%     nAnt = nFeautures;
% end
nAnt = 50;

% NF - the number of features selected by each agent in each cycle.

for c=1:nCycle
    % tic;
    fprintf('cycle : %i \n', c);
    randomAnts = randperm(nFeautures, nAnt);
    FC = zeros(nAnt, nFeautures);
    
    for ant=1:nAnt
%         NF = randi([1, nFeautures]);
        NF = floor(nFeautures / 6);
        % fprintf(' ant : %i / %i \n', ant, nAnt);
        FC(ant, :) = moveAnt(randomAnts(ant), pheromones, ...
            NF, nFeautures, featureCorrelation, featureLabelCorrelation, heuristicMethod);
    end 
    
    
    FC = sum(FC);
    pheromones = ( (1 - decayRate) * pheromones) + ( FC ./ sum(FC) );
    
%     plot(1:nFeautures, pheromones);
%     pause(.05);
%     toc;
end
end

function [featureCounter] = moveAnt (ant, p, NF, nFeautures, featureCorrelation, featureLabelCorrelation, heuristicMethod)    

visitedFeatures = zeros(1, nFeautures);
visitedFeatures(ant) = 1;
featureCounter = zeros(1, nFeautures);
% exploreExploitCoeff = 1;
beta = 1;

explore = 0;
exploit = 0;

for i=1:NF    
    exploreExploitCoeff = 1 * (1 - i/NF)^0.7;
    % fprintf(' ant : %i, NF : %i \n', ant, i);
    if rand() > exploreExploitCoeff
        exploit = exploit + 1;
        if heuristicMethod == 1
            next = heuristic(ant, p, visitedFeatures, featureCorrelation, featureLabelCorrelation, beta);
        else
            next = heuristic2(ant, p, visitedFeatures, featureCorrelation, featureLabelCorrelation, beta);
        end
    else 
        explore = explore + 1;
        next = probability(ant, p, visitedFeatures, featureCorrelation, featureLabelCorrelation, beta);
    end
    
%     fprintf(' i : %i / %i, decay : %i, g=%i, p=%i \n', i, NF, exploreExploitCoeff, exploit, explore);
%     pause(0.1)
    
    visitedFeatures(next) = 1;
    ant = next;
    featureCounter(next) = featureCounter(next) + 1;
    
end
% fprintf('explore : %i, exploit : %i \n', explore, exploit)
% pause(0.1);
end

function [next] = heuristic (ant, p, visited, featureCorrelation, featureLabelCorrelation, beta) 
p(~~visited) = 0;
corre = featureLabelCorrelation;
sim = featureCorrelation(ant, :);
sim = (1 ./ sim) .^ beta ;
[~, next] = max(p .* sim .* corre);
end

function [next] = probability(ant, p, visited, featureCorrelation, featureLabelCorrelation, beta)
p(~~visited) = 0;
corre = featureLabelCorrelation;
sim = featureCorrelation(ant, :);
sim = ( 1 ./  sim) .^ beta;
prod = p .* sim .* corre;
prob = (prod) ./ sum(prod);
cs = cumsum(prob);
next = find(cs>rand,1);
end


