function [partitionindices] = partition(Nsubject,Nsamples)
    num=floor((1/Nsubject) * Nsamples);
    allInd = randperm(Nsamples);
    partitionindices=[];
    for i=1:Nsubject
        I = sort(allInd((i-1)*num+1:i*num));
        partitionindices=[partitionindices;I];
    end
end